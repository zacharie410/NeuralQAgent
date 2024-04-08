local NeuralQAgent = require(script.Parent.NeuralLAgent)

-- Main module logic
local function gameAI()
	math.randomseed(137)  -- Set a fixed seed for reproducibility
	
	local gameAITrainingData = {
		{inputs = {0.9, 0.5, 0.2}, output = {1, 0, 0}},  -- High health, enemy at medium health, health pack close: Attack
		{inputs = {0.3, 0.8, 0.7}, output = {0, 0, 1}},  -- Low health, enemy at high health, health pack far: Retreat
		{inputs = {0.6, 0.6, 0.4}, output = {0, 1, 0}},  -- Equal health, health pack at moderate distance: Defend
		-- Additional scenarios
	}

	
	local nn = NeuralQAgent.LinearNeuralNetwork.new(3, 6, 3)  -- 3 input neurons, 6 hidden neurons, 3 output neurons (Attack, Defend, Retreat)

	-- Train the neural network with the game AI training data
	nn:train(gameAITrainingData, 1000)

	-- Example in-game scenario: AI at 50% health, enemy at 70% health, health pack is nearby
	local currentState = {0.5, 0.7, 0.1}
	local decision = nn:predict(currentState)
	print("AI Decision - Attack:", decision[1], "Defend:", decision[2], "Retreat:", decision[3])

	-- The AI agent chooses the action with the highest score
	local action = "Attack"
	if decision[2] > decision[1] and decision[2] > decision[3] then
		action = "Defend"
	elseif decision[3] > decision[1] and decision[3] > decision[2] then
		action = "Retreat"
	end
	print("AI chooses to:", action)
end

gameAI()
