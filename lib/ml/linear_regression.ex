defmodule ML.LinearRegression do
  use GenServer

  alias ML.LinearRegression.CsvLoader

  require Logger

  defmodule State do
    defstruct batch_size: 100,
              batch_quantity: nil,
              features: nil,
              iterations: 100,
              labels: nil,
              learning_rate: 0.1,
              mean: nil,
              mse_history: [],
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
            mse_history: [float],
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

  def mse_chart do
    GenServer.call(__MODULE__, :mse_chart)
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

    predictions = Nx.dot(test_features, weights)

    # SS_res
    res =
      Nx.subtract(test_labels, predictions)
      |> Nx.pow(2)
      |> Nx.sum()
      |> Nx.to_number()

    # SS_tot
    tot =
      Nx.subtract(test_labels, Nx.mean(test_labels))
      |> Nx.pow(2)
      |> Nx.sum()
      |> Nx.to_number()

    # Coefficient of Determination
    r = 1 - res / tot

    {:reply, r, state}
  end

  def handle_call({:predict, features}, _from, %State{} = state) do
    features
    |> Nx.tensor()
    |> standardize(state.mean, state.std_dev)
    |> process_features()
    |> Nx.dot(state.weights)
    |> Nx.to_list()
    |> then(&{:reply, &1, state})
  end

  def handle_call(:mse_chart, _from, %State{mse_history: mse_history} = state) do
    mse_history
    |> Enum.reverse()
    |> Enum.with_index()
    |> Contex.Dataset.new(["mse", "x"])
    |> Contex.PointPlot.new(mapping: %{x_col: "x", y_cols: ["mse"]})
    |> then(&Contex.Plot.new(600, 400, &1))
    |> Contex.Plot.titles(
      "Mean Squared Error History",
      "Batch size: #{state.batch_size}, iterations: #{state.iterations}"
    )
    |> Contex.Plot.axis_labels("Iteration", "Mean Squared Error")
    |> Contex.Plot.to_svg()
    |> then(
      &File.write("./mse-history-b#{state.batch_size}-i#{state.iterations}.svg", elem(&1, 1))
    )

    {:reply, nil, state}
  end

  @impl true
  def handle_cast({:train, opts}, %State{} = state) do
    state
    |> set_iterations(opts)
    |> set_batching(opts)
    |> Map.put(:training_iteration, 0)
    |> Map.put(:mse_history, [])
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

    mse_history = [calculate_mse(features, weights, labels) | state.mse_history]
    learning_rate = update_learning_rate(mse_history, learning_rate)

    state =
      state
      |> Map.put(:weights, weights)
      |> Map.put(:mse_history, mse_history)
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
  Calculate slope of MSE with respect to weights
  `(features * (features * weights) - label) / n`
  """
  def gradient_descent(features, weights, labels, learning_rate) do
    current_guesses = Nx.dot(features, weights)
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
  Record MSE to adjust the learning rate on the fly
  """
  def calculate_mse(features, weights, labels) do
    features
    |> Nx.dot(weights)
    |> Nx.subtract(labels)
    |> Nx.pow(2)
    |> Nx.sum()
    |> Nx.divide(Nx.shape(features) |> elem(0))
    |> Nx.to_number()
  end

  @doc """
  This function updates `learning_rate` to get closer to the optimal solution.
  The values used for increasing/decreasing `learning_rate` are totally arbitrary.
  """
  def update_learning_rate(mse_history, learning_rate) when length(mse_history) < 2 do
    learning_rate
  end

  def update_learning_rate(mse_history, learning_rate) do
    {[latest, previous], _} = Enum.split(mse_history, 2)

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
