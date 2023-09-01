defmodule ML.LogisticRegression do
  use GenServer

  alias ML.LogisticRegression.CsvLoader

  require Logger

  defmodule State do
    defstruct batch_size: 100,
              batch_quantity: nil,
              features: nil,
              iterations: 100,
              labels: nil,
              learning_rate: 0.1,
              mean: nil,
              cost_history: [],
              standardize_features: true,
              std_dev: nil,
              test_features: nil,
              test_labels: nil,
              test_size: 50,
              training_iteration: nil,
              weights: nil

    @type t :: %State{
            batch_size: integer,
            batch_quantity: integer | nil,
            features: Nx.t() | nil,
            iterations: integer,
            labels: Nx.t() | nil,
            learning_rate: float,
            mean: Nx.t() | nil,
            cost_history: [float],
            standardize_features: boolean,
            std_dev: Nx.t() | nil,
            test_features: Nx.t() | nil,
            test_labels: Nx.t() | nil,
            test_size: integer,
            training_iteration: integer | nil,
            weights: Nx.t() | nil
          }
  end

  # Client

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def load_cars_data do
    GenServer.call(__MODULE__, :load_data)
  end

  def train(opts \\ []) do
    GenServer.cast(__MODULE__, {:train, opts})
  end

  def test do
    GenServer.call(__MODULE__, :test)
  end

  def cost_chart do
    GenServer.call(__MODULE__, :cost_chart)
  end

  @doc """
  Predict MPG for the given features. It accepts a list of list of the following features:
  `[horsepower, cylinders, weight, displacement, modelyear]`
  """
  @spec predict([[number]]) :: list
  def predict(features) do
    GenServer.call(__MODULE__, {:predict, features})
  end

  # Server

  @impl true
  @spec init([]) :: {:ok, State.t()}
  def init(_init_args \\ []) do
    {:ok, %State{}}
  end

  @impl true
  def handle_call(:load_data, _from, %State{} = state) do
    case CsvLoader.load_cars_csv_file(state.test_size) do
      {:ok, {features, labels, test_features, test_labels}} ->
        state
        |> Map.put(:features, Nx.tensor(features))
        |> Map.put(:labels, Nx.tensor(labels))
        |> Map.put(:test_features, Nx.tensor(test_features))
        |> Map.put(:test_labels, Nx.tensor(test_labels))
        |> maybe_standardize_features()
        |> adjust_features()
        |> set_weights()
        |> tap(&Logger.info("Loaded data for #{&1.features.shape |> elem(1)} features"))
        |> then(&{:reply, nil, &1})

      {:error, error} ->
        {:reply, error, state}
    end
  end

  def handle_call(:test, _from, %State{} = state) do
    %{
      test_features: test_features,
      test_labels: test_labels,
      weights: weights
    } = state

    incorrect_count =
      test_features
      |> Nx.dot(weights)
      |> Nx.sigmoid()
      |> Nx.round()
      |> Nx.subtract(test_labels)
      |> Nx.abs()
      |> Nx.sum()
      |> Nx.to_number()

    r = (elem(test_labels.shape, 0) - incorrect_count) / elem(test_labels.shape, 0)

    {:reply, r, state}
  end

  def handle_call({:predict, features}, _from, %State{} = state) do
    features
    |> Nx.tensor()
    |> standardize(state.mean, state.std_dev)
    |> process_features()
    |> Nx.dot(state.weights)
    |> Nx.sigmoid()
    |> Nx.to_list()
    |> then(&{:reply, &1, state})
  end

  def handle_call(:cost_chart, _from, %State{cost_history: cost_history} = state) do
    cost_history
    |> Enum.reverse()
    |> Enum.with_index()
    |> Contex.Dataset.new(["cost", "x"])
    |> Contex.PointPlot.new(mapping: %{x_col: "x", y_cols: ["cost"]})
    |> then(&Contex.Plot.new(600, 400, &1))
    |> Contex.Plot.titles(
      "Cross Entropy History",
      "Batch size: #{state.batch_size}, iterations: #{state.iterations}"
    )
    |> Contex.Plot.axis_labels("Iteration", "Cross Entropy")
    |> Contex.Plot.to_svg()
    |> then(
      &File.write("./cost-history-b#{state.batch_size}-i#{state.iterations}.svg", elem(&1, 1))
    )

    {:reply, nil, state}
  end

  @impl true
  def handle_cast({:train, opts}, %State{} = state) do
    state
    |> set_iterations(opts)
    |> set_batching(opts)
    |> Map.put(:training_iteration, 0)
    |> Map.put(:cost_history, [])
    |> Map.put(:learning_rate, %State{}.learning_rate)
    |> set_weights()
    |> tap(fn _ -> send(self(), :training_iteration) end)
    |> tap(fn _ -> Logger.info("Started training", ansi_color: :green) end)
    |> then(&{:noreply, &1})
  end

  @impl true
  def handle_info(:training_iteration, %State{} = state) do
    %{
      features: features,
      weights: weights,
      labels: labels,
      learning_rate: learning_rate,
      training_iteration: training_iteration
    } = state

    weights = batch_gradient_descent(state, weights, 0)

    cost_history = [calculate_cost(features, weights, labels) | state.cost_history]
    learning_rate = update_learning_rate(cost_history, learning_rate)

    state =
      state
      |> Map.put(:weights, weights)
      |> Map.put(:cost_history, cost_history)
      |> Map.put(:learning_rate, learning_rate)
      |> Map.update(:training_iteration, training_iteration, fn iteration -> iteration + 1 end)

    if state.training_iteration >= state.iterations do
      Logger.info("Finished training", ansi_color: :green)

      state
      |> Map.put(:training_iteration, nil)
      |> then(&{:noreply, &1})
    else
      send(self(), :training_iteration)
      {:noreply, state}
    end
  end

  def batch_gradient_descent(%State{} = state, new_weights, current_iteration)
      when current_iteration < state.batch_quantity do
    %{
      batch_size: batch_size,
      features: features,
      labels: labels,
      learning_rate: learning_rate
    } = state

    start_index = current_iteration * batch_size

    features_slice =
      Nx.slice(features, [start_index, 0], [batch_size, features.shape |> elem(1)])

    labels_slice = Nx.slice(labels, [start_index, 0], [batch_size, labels.shape |> elem(1)])

    updated_weights =
      gradient_descent(
        features_slice,
        new_weights,
        labels_slice,
        learning_rate
      )

    batch_gradient_descent(state, updated_weights, current_iteration + 1)
  end

  def batch_gradient_descent(_state, weights, _current_iteration) do
    weights
  end

  @doc """
  Calculate slope of cost with respect to weights
  `(features * (features * weights) - label) / n`
  """
  def gradient_descent(features, weights, labels, learning_rate) do
    current_guesses = Nx.dot(features, weights) |> Nx.sigmoid()
    differences = Nx.subtract(labels, current_guesses)

    gradient_descent =
      features
      |> Nx.transpose()
      |> Nx.dot(differences)
      |> Nx.multiply(-1 / elem(features.shape, 0))

    gradient_descent
    |> Nx.multiply(learning_rate)
    |> then(&Nx.subtract(weights, &1))
  end

  @doc """
  Record cost to adjust the learning rate on the fly
  -1/a * ((Y^T * log(Y') + (1-Y)^T * log(1-Y'))
  """
  def calculate_cost(features, weights, labels) do
    guesses =
      features
      |> Nx.dot(weights)
      |> Nx.sigmoid()

    term_one =
      labels
      |> Nx.transpose()
      |> Nx.dot(Nx.log(guesses))

    term_two =
      labels
      |> Nx.multiply(-1)
      |> Nx.add(1)
      |> Nx.transpose()
      |> Nx.dot(
        guesses
        |> Nx.multiply(-1)
        |> Nx.add(1)
        |> Nx.log()
      )

    term_one
    |> Nx.add(term_two)
    |> Nx.divide(elem(features.shape, 0))
    |> Nx.multiply(-1)
    |> Nx.squeeze()
    |> Nx.to_number()
  end

  @doc """
  This function updates `learning_rate` to get closer to the optimal solution.
  The values used for increasing/decreasing `learning_rate` are totally arbitrary.
  """
  def update_learning_rate(cost_history, learning_rate) when length(cost_history) < 2 do
    learning_rate
  end

  def update_learning_rate(cost_history, learning_rate) do
    {[latest, previous], _} = Enum.split(cost_history, 2)

    if latest > previous do
      learning_rate / 2
    else
      learning_rate * 1.05
    end
  end

  @doc """
    Standardize `features`, `test_features` and set `mean` and `std_dev`  current_iteration < iterations the `standardize_features` flag is set.
  """
  def maybe_standardize_features(%State{standardize_features: false} = state) do
    state
  end

  def maybe_standardize_features(%State{standardize_features: true, features: features} = state) do
    mean = Nx.mean(features, axes: [0])
    std_dev = Nx.standard_deviation(features, axes: [0])

    state
    |> Map.put(:mean, Nx.mean(features, axes: [0]))
    |> Map.put(:std_dev, Nx.standard_deviation(features, axes: [0]))
    |> Map.update(:features, features, &standardize(&1, mean, std_dev))
    |> Map.update(
      :test_features,
      state.test_features,
      &standardize(&1, mean, std_dev)
    )
  end

  def adjust_features(%State{} = state) do
    state
    |> Map.update(:features, state.features, &process_features/1)
    |> Map.update(:test_features, state.test_features, &process_features/1)
  end

  @spec process_features(Nx.t()) :: Nx.t()
  defp process_features(features) do
    ones = Nx.broadcast(1, {elem(Nx.shape(features), 0), 1})
    Nx.concatenate([ones, features], axis: 1)
  end

  def set_weights(%State{features: features} = state) do
    init_weights = Nx.broadcast(0, {features.shape |> elem(1), 1})
    %State{state | weights: init_weights}
  end

  @doc """
  Standardize given features
  """
  def standardize(features, mean, std_dev) do
    features
    |> Nx.subtract(mean)
    |> Nx.divide(std_dev)
  end

  defp set_iterations(%State{} = state, opts) do
    %State{state | iterations: Keyword.get(opts, :iterations, state.iterations)}
  end

  defp set_batching(%State{batch_size: batch_size} = state, opts) do
    batch_size = Keyword.get(opts, :batch_size, batch_size)
    examples_count = state.features.shape |> elem(0)

    batch_size =
      if batch_size > examples_count do
        Logger.warning(
          "batch_size exceeds examples count, setting it to the count #{examples_count}"
        )

        examples_count
      else
        batch_size
      end

    state
    |> Map.put(:batch_size, batch_size)
    |> Map.put(:batch_quantity, floor(elem(state.features.shape, 0) / batch_size))
  end
end

# weights = Nx.tensor([
#   [1],
#   [1]
# ])

# features = Nx.tensor([
#   [1, 40],
#   [1, 95],
#   [1, 120],
#   [1, 130],
#   [1, 175],
#   [1, 300],
# ])

# features |> Nx.dot(weights) |> Nx.sigmoid()
