local NeuralQAgent = require(script.Parent.NeuralQAgent)

-- Main module logic
local function main()
	math.randomseed(137)  -- Set a fixed seed for reproducibility

	-- Initialize the neural network
	local nn = NeuralQAgent.LinearNeuralNetwork.new(3, 6, 1)  -- 3 input features, 6 hidden neurons, 1 output

	-- Simulated training data representing [Credit Score, Income-to-Debt Ratio, Years of Credit History]
	local trainingData = {
		{inputs = {0.65, 0.5, 0.15}, output = {0.1}},  -- Low risk
		{inputs = {0.55, 0.7, 0.05}, output = {0.5}},  -- Medium risk
		{inputs = {0.30, 0.8, 0.02}, output = {0.9}},  -- High risk
		-- Additional data would be added here in a real scenario
	}

	-- Train the neural network
	local epochs = 1000
	nn:train(trainingData, epochs)

	-- Example prediction: New customer data
	local newCustomer = {0.6, 0.4, 0.1}  -- [Credit Score, Income-to-Debt Ratio, Years of Credit History]
	local prediction = nn:predict(newCustomer)
	print("Likelihood of default for new customer:", table.unpack(prediction))

	-- Optional: Find an optimal customer profile that minimizes default risk
	local startInput = {0.5, 0.5, 0.1}  -- Starting point for optimization
	local optimizedInput = nn:simulatedAnnealing(startInput, 500, 10, 0.7)
	local optimizedPrediction = nn:predict(optimizedInput)
	print("Optimized Customer Profile:", table.unpack(optimizedInput))
	print("Optimized Default Likelihood:", table.unpack(optimizedPrediction))
end

main()
