defmodule ML.LinearRegression do
  use GenServer

  alias ML.LinearRegression.CsvLoader

  require Logger

  defmodule State do
    defstruct learning_rate: 0.1,
              iterations: 100,
              mse_history: [],
              weights: nil,
              features: nil,
              mean: nil,
              std_dev: nil,
              standardize_features: true,
              labels: nil,
              test_size: 50,
              test_features: nil,
              test_labels: nil,
              training_iteration: nil

    @type t :: %State{
            learning_rate: float,
            iterations: integer,
            mse_history: [float],
            weights: Nx.t() | nil,
            features: Nx.t() | nil,
            mean: Nx.t() | nil,
            std_dev: Nx.t() | nil,
            standardize_features: boolean,
            labels: Nx.t() | nil,
            test_size: integer,
            test_features: Nx.t() | nil,
            test_labels: Nx.t() | nil,
            training_iteration: integer | nil
          }
  end

  # Client

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def load_cars_data do
    GenServer.call(__MODULE__, :load_data)
  end

  def train do
    GenServer.cast(__MODULE__, :train)
  end

  def test do
    GenServer.call(__MODULE__, :test)
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
  @spec init(keyword) :: {:ok, State.t()}
  def init(opts \\ []) do
    state =
      %State{}
      |> set_learning_rate(opts)
      |> set_iterations(opts)
      |> set_standardize_features(opts)

    {:ok, state}
  end

  defp set_learning_rate(%State{} = state, opts) do
    %State{state | learning_rate: Keyword.get(opts, :learning_rate, state.learning_rate)}
  end

  defp set_iterations(%State{} = state, opts) do
    %State{state | iterations: Keyword.get(opts, :iterations, state.iterations)}
  end

  defp set_standardize_features(%State{} = state, opts) do
    %State{
      state
      | standardize_features: Keyword.get(opts, :standardize_features, state.standardize_features)
    }
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
    %{test_features: test_features, test_labels: test_labels, weights: weights} = state

    predictions = Nx.dot(test_features, weights)

    # SSres
    res =
      Nx.subtract(test_labels, predictions)
      |> Nx.pow(2)
      |> Nx.sum()
      |> Nx.to_number()

    # SStot
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

  @impl true
  def handle_cast(:train, %State{} = state) do
    state
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

    weights = gradient_descent(features, weights, labels, learning_rate)
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

  @doc """
  Calculate slope of MSE with respect to weights
  `(features * (features * weights) - label) / n`
  """
  def gradient_descent(features, weights, labels, learning_rate) do
    current_guesses = Nx.dot(features, weights)
    differences = Nx.subtract(current_guesses, labels)

    slopes =
      features
      |> Nx.transpose()
      |> Nx.dot(differences)
      |> Nx.divide(features.shape |> elem(0))

    slopes
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
    Standardize `features`, `test_features` and set `mean` and `std_dev` when the `standardize_features` flag is set.
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
  def process_features(features) do
    ones = Nx.broadcast(1, {Nx.shape(features) |> elem(0), 1})
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
end

# use
# {:ok, _pid} = LinearRegression.start_link()
# LinearRegression.load_cars_data()
# LinearRegression.train()
# LinearRegression.test()

# features = [
#   [175, 8, 2.211, 400, 72],
#   [120, 4, 1.4895, 120, 72],
#   [82, 4, 1.36, 119, 82],
#   [193, 8, 2.366, 304, 70],
#   [48, 4, 1.0425, 90, 80]
# ]

# LinearRegression.predict(features)
