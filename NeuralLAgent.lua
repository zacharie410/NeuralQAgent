--.luau ModuleScript

-- Utility Functions
local function sigmoid(x)
	return 1 / (1 + math.exp(-x))
end

local function dSigmoid(y)  -- Derivative of sigmoid
	return y * (1 - y)
end

local function initializeWeights(neuronsA, neuronsB)
	local weights = {}
	for i = 1, neuronsA do
		weights[i] = {}
		for j = 1, neuronsB do
			weights[i][j] = math.sqrt(1 / neuronsB) * (math.random() * 2 - 1)
		end
	end
	return weights
end

local function initializeBiases(neurons)
	local biases = {}
	for i = 1, neurons do
		biases[i] = math.sqrt(1 / neurons) * (math.random() * 2 - 1)
	end
	return biases
end

-- Base Neural Network Module
local BaseNeuralNetwork = {}
BaseNeuralNetwork.__index = BaseNeuralNetwork

function BaseNeuralNetwork.new(inputNeurons, hiddenNeurons, outputNeurons, weightInitializer, biasInitializer)
	local self = setmetatable({}, BaseNeuralNetwork)
	self.weightsInputHidden = weightInitializer(inputNeurons, hiddenNeurons)
	self.weightsHiddenOutput = weightInitializer(hiddenNeurons, outputNeurons)
	self.biasHidden = biasInitializer(hiddenNeurons)
	self.biasOutput = biasInitializer(outputNeurons)
	self.learningRate = 0.01
	self.decay = 0.001
	return self
end

-- Shared methods like forwardPass, train, incrementalTrain, backwardPass go here
function BaseNeuralNetwork:incrementalTrain(input, targetOutput)
	-- Forward pass
	local hiddenActivations = self:forwardPass(input, self.weightsInputHidden, self.biasHidden, sigmoid)
	local predictedOutputs = self:forwardPass(hiddenActivations, self.weightsHiddenOutput, self.biasOutput, sigmoid)

	-- Calculate error
	local errors = {}
	for i = 1, #predictedOutputs do
		errors[i] = targetOutput[i] - predictedOutputs[i]
	end

	-- Backward pass (adjust weights and biases)
	self:backwardPass(input, hiddenActivations, predictedOutputs, errors)
end

function BaseNeuralNetwork:backwardPass(inputs, hiddenActivations, predictedOutputs, errors)
	-- Calculate gradient for weights between hidden and output layer
	local dWeightsHiddenOutput = {}
	for j = 1, #self.weightsHiddenOutput do
		dWeightsHiddenOutput[j] = {}
		for k = 1, #predictedOutputs do
			dWeightsHiddenOutput[j][k] = errors[k] * dSigmoid(predictedOutputs[k]) * hiddenActivations[j]
		end
	end

	-- Adjust weights and biases for hidden-output layer
	for j = 1, #self.weightsHiddenOutput do
		for k = 1, #predictedOutputs do
			self.weightsHiddenOutput[j][k] = self.weightsHiddenOutput[j][k] + self.learningRate * dWeightsHiddenOutput[j][k]
		end
	end
	for k = 1, #self.biasOutput do
		self.biasOutput[k] = self.biasOutput[k] + self.learningRate * errors[k] * dSigmoid(predictedOutputs[k])
	end

	-- Calculate gradient for weights between input and hidden layer (backpropagate the errors)
	local dWeightsInputHidden = {}
	for i = 1, #self.weightsInputHidden do
		dWeightsInputHidden[i] = {}
		for j = 1, #hiddenActivations do
			local errorSum = 0
			for k = 1, #predictedOutputs do
				errorSum = errorSum + errors[k] * dSigmoid(predictedOutputs[k]) * self.weightsHiddenOutput[j][k]
			end
			dWeightsInputHidden[i][j] = errorSum * dSigmoid(hiddenActivations[j]) * inputs[i]
		end
	end

	-- Adjust weights and biases for input-hidden layer
	for i = 1, #self.weightsInputHidden do
		for j = 1, #hiddenActivations do
			self.weightsInputHidden[i][j] = self.weightsInputHidden[i][j] + self.learningRate * dWeightsInputHidden[i][j]
		end
	end
	for j = 1, #self.biasHidden do
		local errorSum = 0
		for k = 1, #predictedOutputs do
			errorSum = errorSum + errors[k] * dSigmoid(predictedOutputs[k]) * self.weightsHiddenOutput[j][k]
		end
		self.biasHidden[j] = self.biasHidden[j] + self.learningRate * errorSum * dSigmoid(hiddenActivations[j])
	end
end

function BaseNeuralNetwork:forwardPass(inputs, weights, biases, activationFunc)
	local outputs = {}
	for i = 1, #biases do
		outputs[i] = biases[i]
		for j = 1, #inputs do
			outputs[i] = outputs[i] + inputs[j] * weights[j][i]
		end
		outputs[i] = activationFunc(outputs[i])
	end
	return outputs
end

function BaseNeuralNetwork:predict(inputs)
	local hiddenActivations = {}
	for i = 1, #self.biasHidden do
		hiddenActivations[i] = 0
		for j = 1, #inputs do
			hiddenActivations[i] = hiddenActivations[i] + inputs[j] * self.weightsInputHidden[j][i]
		end
		hiddenActivations[i] = sigmoid(hiddenActivations[i] + self.biasHidden[i])
	end

	local outputActivations = {}
	for i = 1, #self.biasOutput do
		outputActivations[i] = 0
		for j = 1, #hiddenActivations do
			outputActivations[i] = outputActivations[i] + hiddenActivations[j] * self.weightsHiddenOutput[j][i]
		end
		outputActivations[i] = sigmoid(outputActivations[i] + self.biasOutput[i])
	end
	return outputActivations
end

function BaseNeuralNetwork:train(trainingData, epochs)
	for epoch = 1, epochs do
		local totalError = 0
		for _, dataPoint in ipairs(trainingData) do
			local inputs = dataPoint.inputs
			local targetOutputs = dataPoint.output

			-- Forward pass
			local hiddenActivations = self:forwardPass(inputs, self.weightsInputHidden, self.biasHidden, sigmoid)
			local predictedOutputs = self:forwardPass(hiddenActivations, self.weightsHiddenOutput, self.biasOutput, sigmoid)

			-- Calculate error
			local errors = {}
			for i = 1, #predictedOutputs do
				errors[i] = targetOutputs[i] - predictedOutputs[i]
				totalError = totalError + errors[i] ^ 2
			end

			-- Backward pass (adjust weights and biases)
			self:backwardPass(inputs, hiddenActivations, predictedOutputs, errors)
		end

		-- Optional: Print average error every epoch or at certain intervals
		print("Epoch:", epoch, "Average Error:", totalError / #trainingData)
	end
end

-- Linear Neural Network Module
local LinearNeuralNetwork = setmetatable({}, {__index = BaseNeuralNetwork})

function LinearNeuralNetwork.new(inputNeurons, hiddenNeurons, outputNeurons)
	local self = BaseNeuralNetwork.new(inputNeurons, hiddenNeurons, outputNeurons, initializeWeights, initializeBiases)
	setmetatable(self, {__index = LinearNeuralNetwork})  -- Set the metatable to NeuralNetwork to access its methods
	return self
end

function LinearNeuralNetwork:simulatedAnnealing(startInput, iterations, temp, coolingRate)
	local function tweakInput(input, tweakMagnitude)
		local tweakedInput = {}
		for i = 1, #input do
			-- Tweak each input within the range [0, 1] as an example
			tweakedInput[i] = math.max(0, math.min(1, input[i] + (math.random() - 0.5) * tweakMagnitude))
		end
		return tweakedInput
	end

	local currentInput = startInput
	local currentScore = self:predict(currentInput)[1]  -- Assuming single output for simplicity
	local bestInput = currentInput
	local bestScore = currentScore

	for i = 1, iterations do
		local tweakMagnitude = temp / 10  -- Adaptive tweaking based on temperature
		local newInput = tweakInput(currentInput, tweakMagnitude)
		local newScore = self:predict(newInput)[1]  -- Assuming single output for simplicity

		local delta = newScore - currentScore

		if delta > 0 or math.random() < math.exp(-delta / temp) then
			currentInput = newInput
			currentScore = newScore

			if newScore > bestScore then
				bestInput = newInput
				bestScore = newScore
			end
		end

		temp = temp * coolingRate
	end

	return bestInput
end

-- Data Management Module
local DataModule = {}
DataModule.__index = DataModule

function DataModule.new()
	local self = setmetatable({}, DataModule)
	self.trainingData = {}
	return self
end

function DataModule:addData(inputs, output)
	table.insert(self.trainingData, {inputs = inputs, output = output})
end

--Export Neural Network

return {LinearNeuralNetwork = LinearNeuralNetwork}