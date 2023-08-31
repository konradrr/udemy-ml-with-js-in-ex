# Machine Learning with JavaScript (in Elixir :) 

This repo contains my learning progress in [this course](https://www.udemy.com/course/machine-learning-with-javascript/) from Stephen Grider.

JS is okay, but Elixir is my #1 ❤️ so decided to "translate" JS to Ex on the fly 😉

## Usage

```elixir
  # Start LinearRegression server
  {:ok, _pid} = LinearRegression.start_link()

  # Load cars data (hardcoded for now)
  LinearRegression.load_cars_data()

  # Train the model
  LinearRegression.train()

  # Test the model and return Coefficient of Determination
  LinearRegression.test()

  # Predict MPG for the given features sets (`[horsepower, cylinders, weight, displacement, modelyear]`)
  features = [
    [175, 8, 2.211, 400, 72],
    [120, 4, 1.4895, 120, 72],
    [82, 4, 1.36, 119, 82],
    [193, 8, 2.366, 304, 70],
    [48, 4, 1.0425, 90, 80]
  ]

  LinearRegression.predict(features)
```