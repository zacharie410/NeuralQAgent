local NeuralQAgent = require(script.Parent.NeuralQAgent)

-- Main module logic
local function main()
	math.randomseed(137)  -- For reproducibility
	
	--{inputs = {Feature1, Feature2, Feature3}, output = {Explore, Support, Combat, Retreat}}

	local trainingData = {
		-- Enemy close, few resources, little support: Retreat
		{inputs = {0.9, 0.2, 0.1}, output = {0, 0, 0, 1}},

		-- Enemy far, abundant resources, moderate support: Explore
		{inputs = {0.1, 0.8, 0.5}, output = {1, 0, 0, 0}},

		-- Enemy medium, moderate resources, strong support: Call for Support
		{inputs = {0.5, 0.5, 0.9}, output = {0, 1, 0, 0}},

		-- Enemy close, abundant resources, little support: Engage in Combat
		{inputs = {0.7, 0.8, 0.2}, output = {0, 0, 1, 0}},

		-- Enemy far, few resources, strong support: Explore/Call for Support
		{inputs = {0.3, 0.1, 0.8}, output = {0.5, 0.5, 0, 0}},

		-- More nuanced case: Enemy medium, resources medium, support medium
		{inputs = {0.5, 0.5, 0.5}, output = {0.3, 0.3, 0.4, 0}},

		-- Ambiguous case: Enemy medium, few resources, no support
		{inputs = {0.5, 0.2, 0}, output = {0.2, 0.2, 0.3, 0.3}},

		-- Ideal scenario: Enemy far, resources abundant, strong support
		{inputs = {0.1, 1.0, 1.0}, output = {1, 0, 0, 0}}
	}

    -- Initialize a quantum-inspired neural network
	local aiDrone = NeuralQAgent.QuantumNeuralNetwork.new(3, 12, 4)  -- 3 inputs, 6 hidden neurons, 4 outputs (Explore, Support, Combat, Retreat)
	print(aiDrone)
	-- Assume 'trainingData' is adjusted for 4 output neurons, representing different actions
	aiDrone:train(trainingData, 1000)

	local gameScenario = {1, 1, 0}  -- Example scenario
	local actionIndex = aiDrone:quantumInspiredPredict(gameScenario)
	print(actionIndex)
	-- Map the output neuron index to an action
	local actions = {"Explore", "Call for Support", "Engage in Combat", "Retreat"}
	local chosenAction = actions[actionIndex]

	print("AI Decision based on Quantum-Inspired Prediction:", chosenAction)
end

main()
