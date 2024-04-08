# NeuralQ Agent: Quantum-Inspired Metaheuristic Neural Networks for Lua

NeuralQ Agent is a Lua library designed to facilitate the integration of artificial intelligence and neural networks into your projects. This library is particularly tailored for applications ranging from education to game development, offering both classical and quantum-inspired neural network models.

## Key Features

- Simple yet powerful neural network implementation in Lua
- Quantum-inspired algorithms for advanced problem-solving
- Suitable for educational purposes and complex AI in games
- Detailed documentation to help you get started quickly

## Getting Started

To begin using NeuralQ Agent, include the library in your Lua project. This library is designed to be intuitive, allowing you to easily create and train neural networks.

### Example: AI Drone Decision Making

This example demonstrates how to use NeuralQ Agent to simulate an AI drone making strategic decisions based on its environment and status. The scenario considers factors like enemy proximity, resource availability, and support level to decide between exploring, calling for support, engaging in combat, or retreating.

```lua
local NeuralQAgent = require(script.Parent.NeuralQAgent)

-- Main module logic
local function main()
	math.randomseed(137)  -- For reproducibility
	
	local trainingData = {
		-- Enemy close, few resources, little support: Retreat
		{inputs = {0.9, 0.2, 0.1}, output = {0, 0, 0, 1}},
		-- Enemy far, abundant resources, moderate support: Explore
		{inputs = {0.1, 0.8, 0.5}, output = {1, 0, 0, 0}},
		-- Additional scenarios...
	}

    -- Initialize a quantum-inspired neural network
	local aiDrone = NeuralQAgent.QuantumNeuralNetwork.new(3, 12, 4)
	aiDrone:train(trainingData, 1000)

	local gameScenario = {1, 1, 0}  -- Example scenario
	local actionIndex = aiDrone:quantumInspiredPredict(gameScenario)
	local actions = {"Explore", "Call for Support", "Engage in Combat", "Retreat"}
	local chosenAction = actions[actionIndex]

	print("AI Decision:", chosenAction)
end

main()
```

## Documentation

### Neural Networks

- **Linear Neural Network**: Traditional neural network model for a wide range of applications.
- **Quantum Neural Network**: Incorporates quantum-inspired elements for enhanced problem-solving.

### Training and Prediction

- Train your network with `:train(trainingData, epochs)`, where `trainingData` includes input-output pairs and `epochs` specifies the number of training iterations.
- Use `:predict(inputs)` for classical prediction or `:quantumInspiredPredict(inputs)` for quantum-inspired prediction to evaluate new data.

### Quantum-Inspired Techniques

Explore heuristic quantum algorithms with `:quantumInspiredAnnealing(startInput, iterations, temp, coolingRate)` to optimize solutions for complex problems.

## Community and Support

Join our community to share your projects, get help, and contribute to the NeuralQ Agent library. Whether you're a beginner or an expert, your input is valuable in making AI more accessible in the Lua ecosystem.

## License

NeuralQ Agent is open source under the Apache 2.0 License
