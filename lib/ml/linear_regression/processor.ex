defmodule ML.LinearRegression.Processor do
  use GenServer

  alias ML.LinearRegression.CsvLoader

  defmodule State do
    defstruct learning_rate: 0.1,
              iterations: 100,
              mse_history: [],
              weights: [],
              features: [],
              labels: [],
              test_size: 50,
              test_features: [],
              test_labels: []

    @type t :: %State{
            learning_rate: float,
            iterations: integer,
            mse_history: [Nx.t()],
            weights: [Nx.t()],
            labels: [Nx.t()],
            test_size: integer,
            test_features: [Nx.t()],
            test_labels: [Nx.t()]
          }
  end

  # Client

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def load_cars_data(pid) do
    GenServer.call(pid, :load_data)
  end

  # Server

  @impl true
  @spec init(keyword) :: {:ok, State.t()}
  def init(opts \\ []) do
    state =
      %State{}
      |> set_learning_rate(opts)
      |> set_iterations(opts)

    {:ok, state}
  end

  defp set_learning_rate(%State{} = state, opts) do
    %State{state | learning_rate: Keyword.get(opts, :learning_rate, state.learning_rate)}
  end

  defp set_iterations(%State{} = state, opts) do
    %State{state | iterations: Keyword.get(opts, :iterations, state.iterations)}
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
        |> then(&{:reply, "Loaded data for #{&1.features.shape |> elem(1)} features", &1})

      {:error, error} ->
        {:reply, error, state}
    end
  end
end

# {:ok, pid} = LRProcessor.start_link()
# LRProcessor.load_cars_data(LRProcessor)
