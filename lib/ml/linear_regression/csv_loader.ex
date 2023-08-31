defmodule ML.LinearRegression.CsvLoader do
  require Logger
  alias NimbleCSV.RFC4180, as: CSV

  # defmodule ColumnsStruct do

  # end

  @spec load_cars_csv_file(integer) ::
          {:ok, {list, list, list, list}} | {:error, binary}
  def load_cars_csv_file(test_size) do
    root_path = Path.expand(".")
    file_path = Path.join([root_path, "data", "cars.csv"])

    if File.exists?(file_path) do
      {test_data, data} =
        file_path
        |> File.stream!()
        |> CSV.parse_stream()
        |> Stream.map(fn row ->
          [
            _passedemissions,
            mpg,
            cylinders,
            displacement,
            horsepower,
            weight,
            _acceleration,
            modelyear,
            _carname
          ] = row

          {[
             parse_float(horsepower),
             parse_float(cylinders),
             parse_float(weight),
             parse_float(displacement),
             parse_float(modelyear)
           ], [parse_float(mpg)]}
        end)
        |> Enum.to_list()
        |> Enum.shuffle()
        |> Enum.split(test_size)

      features = Enum.map(data, &elem(&1, 0))
      labels = Enum.map(data, &elem(&1, 1))
      test_features = Enum.map(test_data, &elem(&1, 0))
      test_labels = Enum.map(test_data, &elem(&1, 1))

      {:ok, {features, labels, test_features, test_labels}}
    else
      {:error, "File doesn't exist!"}
    end
  end

  defp parse_float(string_float) do
    string_float |> Float.parse() |> elem(0)
  end
end
