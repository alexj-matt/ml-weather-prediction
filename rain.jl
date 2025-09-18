using Markdown
using InteractiveUtils



### activate Packages
begin
    using Pkg
	Pkg.activate(".")
	# Add packages to existing package pool with: Pkg.add("<Package Name>") and Pkg.instantiate (changes Manifest ?or? Project file, only needs to be done once)
    using MLCourse, Plots, MLJ, DataFrames, Random, CSV, Flux, Distributions,
          StatsPlots, MLJFlux, OpenML, Random, MLJLinearModels, Polynomials, Hyperopt
    Core.eval(Main, :(using MLJ)) # hack to make @pipeline work in notebooks
end



### read & treat data
begin
	training_data = CSV.read(joinpath(@__DIR__, "res", "trainingdata.csv"), DataFrame)
	test_data = CSV.read(joinpath(@__DIR__, "res", "testdata.csv"), DataFrame)

	# display data
	describe(training_data,cols=:)
	print("Missing values training_data: ", sum(describe(training_data).nmissing), "\n")
	print("Mean values training_data: ", sum(describe(training_data).mean), "\n\n")

	print("Missing values test_data: ", sum(describe(test_data).nmissing), "\n")
	print("Mean values test_data: ", sum(describe(test_data).mean), "\n\n")

	# get predictors
	training_data_predictor = training_data[:,1:end-1]

	# treat missing data -> fill with mean value of column
	training_data_clean = MLJ.transform(fit!(machine(FillImputer(), training_data_predictor)), training_data_predictor)

	# standardize predictors to variance = 1 and mean = 0 in order to avoid impact of different units etc
	function standardize(x) # only columns with std(col) != 0 get here for example not test_data[;,124]
		return (x .- mean(x)) ./ std(x)
	end

	for pred in (size(training_data_clean)[2]:-1:1)
		if(std(training_data_clean[:,pred]) == 0 || std(test_data[:,pred]) == 0)
			print(pred)
			select!(test_data, Not(pred))
			select!(training_data_clean, Not(pred))
		else
			training_data_clean[:,pred] = Float32.(standardize(training_data_clean[:,pred]))
			test_data[:,pred] = Float32.(standardize(test_data[:,pred]))
		end
	end

	# put back ground truth
	training_data_clean.precipitation = training_data[:,end]

	#to be able to work with multiclass DF
	data_trng = coerce(training_data_clean,:precipitation=>Multiclass) #to work with booleans

	print("Missing values training_data cleaned and normalized: ", sum(describe(training_data_clean).nmissing), "\n")
	print("Missing values test_data normalized: ", sum(describe(test_data).nmissing), "\n")
	describe(training_data_clean,cols=:)
	describe(test_data,cols=:)
end

@df data_trng corrplot([:ABO_radiation_1 :ABO_wind_direction_1 :ENG_delta_pressure_1 :LUG_air_temp_1], labelfontsize = 7, tickfontsize = 6,  size = (700, 700))



### define & fit machines and compute outpute


## Machine 1
#fit mach_1 (logistic classifier)
begin
	Random.seed!(42)
	mach_1 = machine(LogisticClassifier(penalty=:none), data_trng[:,1:end-1],data_trng.precipitation)|>fit!
	MLJ.save(joinpath(@__DIR__, "machines", "mach_1.jlso"), mach_1)
end

# predict mach_1 on test set
begin
	mach_1 = machine(joinpath(@__DIR__, "machines", "mach_1.jlso"))
	# predict using fitted machine
	prediction_1 = predict(mach_1,test_data)
	# create output vector with probability predictions
	output_1 = pdf.(prediction_1,true)
	# create enumerator with length of observations to be predicted
	enumerator = [i for i in 1:length(test_data[1:end,1])]
	# create DataFrame with enumerated probabilities
	outputDF_1 = DataFrame(id = enumerator,precipitation_nextday = output_1)
	# write to output file
	CSV.write( (joinpath(@__DIR__, "out", "output_1.csv")), outputDF_1)
end


## Machine 2
#fit mach_2 (multilayer perceptrons)
begin
	Random.seed!(42)
	builder = MLJFlux.MLP(hidden=(64,64,64), σ = relu)
	model_2 =	NeuralNetworkClassifier( 	builder = builder,
											optimiser = ADAMW(),
											batch_size = 32,
											epochs = 1000)
	mach_2 = fit!(machine(model_2, data_trng[:,1:end-1], data_trng[:,end]))
	MLJ.save(joinpath(@__DIR__, "machines", "mach_2.jlso"), mach_2)
end
# possible changes: NeuralNetworkClassifier -> NeuralNetworkRegressor
#					MLJFlux.MLP				-> MLJFlux.Short
#					ADAMW()					-> ADAM()

# predict mach_2 on test set
begin
	mach_2 = machine(joinpath(@__DIR__, "machines", "mach_2.jlso"))
	prediction_2 = predict(mach_2, test_data)
	output_2 = pdf.(prediction_2, true)
	enumerator = [i for i in 1:length(test_data[1:end,1])]
	outputDF_2 = DataFrame(id = enumerator,precipitation_nextday = output_2)
	CSV.write((joinpath(@__DIR__, "out", "output_2.csv")), outputDF_2)
end


## Machine 3
#fit mach_3 (multilayer perceptrons with ADAM() instead of ADAMW())
begin
	Random.seed!(42)
	builder = MLJFlux.MLP(hidden=(64,64,64), σ = relu)
	model_3 =	NeuralNetworkClassifier( 	builder = builder,
											optimiser = ADAM(),
											batch_size = 32,
											epochs = 1000)
	mach_3 = fit!(machine(model_3, data_trng[:,1:end-1], data_trng[:,end]))
	MLJ.save(joinpath(@__DIR__, "machines", "mach_3.jlso"), mach_3)
end

# predict mach_3 on test set
begin
	mach_3 = machine(joinpath(@__DIR__, "machines", "mach_3.jlso"))
	prediction_3 = predict(mach_3, test_data)
	output_3 = pdf.(prediction_3, true)
	enumerator = [i for i in 1:length(test_data[1:end,1])]
	outputDF_3 = DataFrame(id = enumerator,precipitation_nextday = output_3)
	CSV.write((joinpath(@__DIR__, "out", "output_3.csv")), outputDF_3)
end


## Machine 4
#fit mach_4 (epochs = 2000 instead of epochs = 1000)
begin
	Random.seed!(42)
	builder = MLJFlux.MLP(hidden=(64,64,64), σ = relu)
	model_4 =	NeuralNetworkClassifier( 	builder = builder,
											optimiser = ADAMW(),
											batch_size = 32,
											epochs = 2000)
	mach_4 = fit!(machine(model_4, data_trng[:,1:end-1], data_trng[:,end]))
	MLJ.save(joinpath(@__DIR__, "machines", "mach_4.jlso"), mach_4)
end

# predict mach_4 on test set
begin
	mach_4 = machine(joinpath(@__DIR__, "machines", "mach_4.jlso"))
	prediction_4 = predict(mach_4, test_data)
	output_4 = pdf.(prediction_4, true)
	enumerator = [i for i in 1:length(test_data[1:end,1])]
	outputDF_4 = DataFrame(id = enumerator,precipitation_nextday = output_4)
	CSV.write((joinpath(@__DIR__, "out", "output_4.csv")), outputDF_4)
end


## Machine 5
#fit mach_5 (6 layers with 32 neurons instead of 3 layers with 64 neurons)
begin
	Random.seed!(42)
	builder = MLJFlux.MLP(hidden=(32,32,32,32,32,32), σ = relu)
	model_5 =	NeuralNetworkClassifier( 	builder = builder,
											optimiser = ADAMW(),
											batch_size = 32,
											epochs = 1000)
	mach_5 = fit!(machine(model_5, data_trng[:,1:end-1], data_trng[:,end]))
	MLJ.save(joinpath(@__DIR__, "machines", "mach_5.jlso"), mach_5)
end

# predict mach_5 on test set
begin
	mach_5 = machine(joinpath(@__DIR__, "machines", "mach_5.jlso"))
	prediction_5 = predict(mach_5, test_data)
	output_5 = pdf.(prediction_5, true)
	enumerator = [i for i in 1:length(test_data[1:end,1])]
	outputDF_5 = DataFrame(id = enumerator,precipitation_nextday = output_5)
	CSV.write((joinpath(@__DIR__, "out", "output_5.csv")), outputDF_5)
end


## Machine 6
#fit mach_6 (3 layers with 128 nodes)
begin
	Random.seed!(42)
	builder = MLJFlux.MLP(hidden=(128, 128, 128), σ = relu)
	model_6 =	NeuralNetworkClassifier( 	builder = builder,
											optimiser = ADAMW(),
											batch_size = 32,
											epochs = 1000)
	mach_6 = fit!(machine(model_6, data_trng[:,1:end-1], data_trng[:,end]))
	MLJ.save(joinpath(@__DIR__, "machines", "mach_6.jlso"), mach_6)
end

# predict mach_6 on test set
begin
	mach_6 = machine(joinpath(@__DIR__, "machines", "mach_6.jlso"))
	prediction_6 = predict(mach_6, test_data)
	output_6 = pdf.(prediction_6, true)
	enumerator = [i for i in 1:length(test_data[1:end,1])]
	outputDF_6 = DataFrame(id = enumerator,precipitation_nextday = output_6)
	CSV.write((joinpath(@__DIR__, "out", "output_6.csv")), outputDF_6)
end


## Machine 7
#fit mach_7 (4 layers with 128 nodes)
begin
	Random.seed!(42)
	builder = MLJFlux.MLP(hidden=(128, 128, 128, 128), σ = relu)
	model_7 =	NeuralNetworkClassifier( 	builder = builder,
											optimiser = ADAMW(),
											batch_size = 32,
											epochs = 1000)
	mach_7 = fit!(machine(model_7, data_trng[:,1:end-1], data_trng[:,end]))
	MLJ.save(joinpath(@__DIR__, "machines", "mach_7.jlso"), mach_7)
end

# predict mach_7 on test set
begin
	mach_7 = machine(joinpath(@__DIR__, "machines", "mach_7.jlso"))
	prediction_7 = predict(mach_7, test_data)
	output_7 = pdf.(prediction_7, true)
	enumerator = [i for i in 1:length(test_data[1:end,1])]
	outputDF_7 = DataFrame(id = enumerator,precipitation_nextday = output_7)
	CSV.write((joinpath(@__DIR__, "out", "output_7.csv")), outputDF_7)
end


## Machine 8
#fit mach_8 (3 layers with 128 nodes, sigmoid function)
begin
	Random.seed!(42)
	builder = MLJFlux.MLP(hidden=(128, 128, 128), σ = sigmoid)
	model_8 =	NeuralNetworkClassifier( 	builder = builder,
											optimiser = ADAMW(),
											batch_size = 32,
											epochs = 1000)
	mach_8 = fit!(machine(model_8, data_trng[:,1:end-1], data_trng[:,end]))
	MLJ.save(joinpath(@__DIR__, "machines", "mach_8.jlso"), mach_8)
end

# predict mach_8 on test set
begin
	mach_8 = machine(joinpath(@__DIR__, "machines", "mach_8.jlso"))
	prediction_8 = predict(mach_8, test_data)
	output_8 = pdf.(prediction_8, true)
	enumerator = [i for i in 1:length(test_data[1:end,1])]
	outputDF_8 = DataFrame(id = enumerator,precipitation_nextday = output_8)
	CSV.write((joinpath(@__DIR__, "out", "output_8.csv")), outputDF_8)
end


## Machine 9
#fit mach_9 (3 layers with 128 nodes, tanh function)
begin
	Random.seed!(42)
	builder = MLJFlux.MLP(hidden=(128, 128, 128), σ = tanh)
	model_9 =	NeuralNetworkClassifier( 	builder = builder,
											optimiser = ADAMW(),
											batch_size = 32,
											epochs = 1000)
	mach_9 = fit!(machine(model_9, data_trng[:,1:end-1], data_trng[:,end]))
	MLJ.save(joinpath(@__DIR__, "machines", "mach_9.jlso"), mach_9)
end

# predict mach_9 on test set
begin
	mach_9 = machine(joinpath(@__DIR__, "machines", "mach_9.jlso"))
	prediction_9 = predict(mach_9, test_data)
	output_9 = pdf.(prediction_9, true)
	enumerator = [i for i in 1:length(test_data[1:end,1])]
	outputDF_9 = DataFrame(id = enumerator,precipitation_nextday = output_9)
	CSV.write((joinpath(@__DIR__, "out", "output_9.csv")), outputDF_9)
end


## Machine 10
#fit mach_10 (3 layers with 64 nodes, softmax function)
begin
	Random.seed!(42)
	builder = MLJFlux.MLP(hidden=(64, 64, 64), σ = x->log(exp(x)+1))
	model_10 =	NeuralNetworkClassifier( 	builder = builder,
											optimiser = ADAMW(),
											batch_size = 32,
											epochs = 1000)
	mach_10 = fit!(machine(model_10, data_trng[:,1:end-1], data_trng[:,end]))
	MLJ.save(joinpath(@__DIR__, "machines", "mach_10.jlso"), mach_10)
end

# predict mach_10 on test set
begin
	mach_10 = machine(joinpath(@__DIR__, "machines", "mach_10.jlso"))
	prediction_10 = predict(mach_10, test_data)
	output_10 = pdf.(prediction_10, true)
	enumerator = [i for i in 1:length(test_data[1:end,1])]
	outputDF_10 = DataFrame(id = enumerator,precipitation_nextday = output_10)
	CSV.write((joinpath(@__DIR__, "out", "output_10.csv")), outputDF_10)
end


## Machine 11
#fit mach_11 (3 layers with 64 nodes, softmax function, introduced lambda of 0.2)
begin
	Random.seed!(42)
	builder = MLJFlux.MLP(hidden=(64, 64, 64), σ = x->log(exp(x)+1))
	model_11 =	NeuralNetworkClassifier( 	builder = builder,
											optimiser = ADAMW(),
											batch_size = 32,
											epochs = 1000,
											lambda = 0.2)
	mach_11 = fit!(machine(model_11, data_trng[:,1:end-1], data_trng[:,end]))
	MLJ.save(joinpath(@__DIR__, "machines", "mach_11.jlso"), mach_11)
end

# predict mach_11 on test set
begin
	mach_11 = machine(joinpath(@__DIR__, "machines", "mach_11.jlso"))
	prediction_11 = predict(mach_11, test_data)
	output_11 = pdf.(prediction_11, true)
	enumerator = [i for i in 1:length(test_data[1:end,1])]
	outputDF_11 = DataFrame(id = enumerator,precipitation_nextday = output_11)
	CSV.write((joinpath(@__DIR__, "out", "output_11.csv")), outputDF_11)
end

# sigmoid, relu, tanh, x->log(exp(x)+1))
plot(sigmoid)
# lambda = 0.1
# ?NeuralNetworkClassifier

############## from here on everything is trashy but might still be useful in the future

############## cross - vali #############
function cross_validation_sets(idx, K)
    n = length(idx)
    r = n ÷ K
    [let idx_valid = idx[(i-1)*r+1:(i == K ? n : i*r)]
         (idx_valid = idx_valid, idx_train = setdiff(idx, idx_valid))
     end
     for i in 1:K]
end

function data_split(data;
	shuffle = false,
	idx_train = 1:200,
	idx_valid = 201:250)
idxs = if shuffle
randperm(size(data, 1))
else
1:size(data, 1)
end
(train = data[idxs[idx_train], :],
valid = data[idxs[idx_valid], :],)
end

function fit_and_evaluate(mach, data)
	(train = rmse(predict(mach, select(data.train, :precipitation_nextday)), data.train.y),
	 valid = rmse(predict(mach, select(data.valid, :precipitation_nextday)), data.valid.y),
	 test = rmse(predict(mach, select(data.test, :precipitation_nextday)), data.test.y))
end

function cross_validation(mach, data; K = 5)
    losses = [fit_and_evaluate(mach, data_split(data; idxs...))
              for idxs in cross_validation_sets(1:250, K)]
    (train = mean(getproperty.(losses, :train)),
     valid = mean(getproperty.(losses, :valid)),
     test = mean(getproperty.(losses, :test)))
end
########### end of cross - vali #########
cross_validation_sets(training_data[:,17], 3)
losses_cv10 = [cross_validation(mach_8,training_data,K = 10)]

MLJ.evaluate!
begin

	function train_nn(layers, nodes, batch_size, epochs, sigma)

	end

	combNum = 6
	ho =	@hyperopt for i=combNum,
		sampler = RandomSampler(), # This is default if none provided
		layers = collect(StepRange(3, Int8(1), 6))
		nodes = collect(StepRange(32, Int8(32), 128)),
		batch_size = [16, 32]
		epochs = [1000, 2000]
		sigma = [relu, softmax]
		# c = exp10.(LinRange(-1,3,1000))
		print(i, "\tLayers: ", layers, "\tNodes: ", nodes, "\tBatch size: ", batch_size, "\tEpochs: ", epochs, "\n")

		@show 
	end
end

#implement K-cross validation
#dont forget to shuffle dataset before creating training and validation set
begin
	K = 5 #typical choices K = 5 or K = 10
	MLJ.MLJBase.train_test_pairs(CV(nfolds =K), 1:size(data_trng[:,1])[1]) #this only gives the different sets in form of indices
end # for the moment this doesnt do a whole lot

collect(StepRange(32, Int8(16), 320))


# 
begin
	data_input = data_trng[:,1:end-1]
	data_output = data_trng[:,end]

	print(typeof(data_input))
	print(typeof(data_output))

	# dense configures the different layers --> first as many inputs as there are predictors are passed to 50 nodes, then output of 50 nodes passed to 1 output (true/false)
	# Chain just chains them together
	predNum = size(data_input[1,1:end])[1]
	nn_3 = Chain(Dense(predNum, 50, relu), Dense(50, 1)) 	# takes predNum-dimensional vector as input
															# outputs values of output nodes in a vector (Vector{Float32})
	function loss(x, y)
		output = nn_1(x)[1,:]
		mean(output .- y) # is this the missclassification rate?
		#m = output[1, :]
		#s = softplus.(output[2, :])
		#mean((m .- y) .^ 2 ./ (2 * s) .+ log.(s))
	end

	opt = ADAMW()
	p = Flux.params(nn_1) # these are the parameters to be adapted (all the parameters in nn_1 are passed to p)
	data = Flux.DataLoader(data_tuple, batchsize = 32)
	for _ in 1:50
		Flux.Optimise.train!(loss, p, data, opt)
	end
	# train on a grid of inputs to see where paramters improve/are optimal