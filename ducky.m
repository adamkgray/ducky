% based on this article
% https://pyimagesearch.com/2021/05/06/backpropagation-from-scratch-with-python/
% accessed February 18th, 2022

classdef ducky < handle
    % custom feedforward neural net

    properties
        Layers
        LearningRate
        Weights
        Activation
    end

    methods

        %%%%%%%%%%%%%%%
        % constructor %
        %%%%%%%%%%%%%%%

        function obj = ducky(layers, learningRate, activation)
            % initialise layers
            obj.Layers = layers;

            % initialise learning rate
            obj.LearningRate = learningRate;

            % initialise activation function
            obj.Activation = activation;

            % initialise weights
            % since weights are in between layers,
            % there will always be L - 1 weight matrixes
            weights = cell(1, length(layers) - 1);

            for i = 1:(length(layers) - 1)

                % NOTES:

                % the computation of the weighted sums from one
                % layer to the next is just matrix multiplication

                % the weights connecting two layers are stored
                % as an NxM matrix where 'N' is the number of nodes
                % in the parent layer and 'M' is the number of nodes
                % in the child layer

                % each column in a weight matrix represents
                % the weights from the parent layer to a single node

                % the result of the matrix multiplication will
                % always be a 1xM matrix - the weighted sums of the
                % parent layer

                % add one node to each weight matrix
                % as a bias node
                n = layers(i) + 1;
                m = layers(i + 1) + 1;

                % the last weight matrix for is the output
                % it doesn't need the bias node
                if i == (length(layers) - 1)
                    m = layers(end);
                end

                % dividing the randomised weight be the
                % square root of the original number
                % of nodes in the layer helps avoid the
                % 'vanishing gradient' problem
                % see this answer on stack exchange:
                % https://stats.stackexchange.com/questions/326710/why-is-weight-initialized-as-1-sqrt-of-hidden-nodes-in-neural-networks
                w = normrnd(0, 1, [n m]) / sqrt(layers(i));

                weights{i} = w;
            end

            obj.Weights = weights;
        end

        %%%%%%%
        % SSE %
        %%%%%%%
        function error = sse(~, x, t)
            error = sum((t - x).^2);
        end

        %%%%%%%
        % MSE %
        %%%%%%%
        function error = mse(~, x, t)
            error = sum((t - x).^2) / length(x);
        end

        %%%%%%%%%%%%
        % accuracy %
        %%%%%%%%%%%%
        function acc = accuracy(~, x, t)
            acc = sum(round(x) == t) / length(x);
        end

        %%%%%%%%%%%%%%%%%%%%%%%
        % activation function %
        %%%%%%%%%%%%%%%%%%%%%%%
        function y = activation(obj, x)

            switch obj.Activation
                case 'sig' % sigmoid
                    y = 1 ./ (1 + exp(-x));
                case 'tanh' % hyperbolic tangent
                    y = tanh(x);
                otherwise % default is sigmoid
                    y = 1 ./ (1 + exp(-x));
            end

        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % derivative of the activation function %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function y = activationDerivative(obj, x)

            switch obj.Activation
                case 'sig' % sigmoid
                    f = 1 ./ (1 + exp(-x));
                    y = f .* (1 - f);
                case 'tanh' % hyberbolic tangent
                    y = 1 - (tanh(x).^2);
                otherwise % default is sigmoid
                    f = 1 ./ (1 + exp(-x));
                    y = f .* (1 - f);

            end

        end

        %%%%%%%%%%%%%%%%%
        % train network %
        %%%%%%%%%%%%%%%%%
        function [net, errors] = train(obj, x, t, epochs)

            % we return the net itself
            net = obj;

            % since our weights include an extra node
            % for the bias, we need to add an extra
            % feature to our data
            % this feature is just a '1' which will
            % then be continually scaled for each layer
            xWithBias = [x ones(size(x, 1), 1)];

            % store the error rate for each epoch
            % in a vector
            errors = zeros(1, epochs);

            % display waitbar
            f = waitbar(0, "");

            % iterate over epochs
            for e = 1:epochs

                % fit training data
                % this means for each sample we
                % conduct a forward pass and
                % then do backpropagation
                for i = 1:size(xWithBias, 1)
                    [weightedSums, activations] = forwardPass(obj, xWithBias(i, :));
                    backpropagation(obj, t(i, :), weightedSums, activations);
                end

                % remember error rate for this epoch
                acc = accuracy(obj, predict(obj, x), t);
                errors(e) = 1 - acc;

                waitbar(e / epochs, f, sprintf('training %d/%d, error rate %f', e, epochs, 1 - acc));

            end

            close(f)

        end

        %%%%%%%%%%%%%%%%
        % forward pass %
        %%%%%%%%%%%%%%%%
        function [weightedSums, activations] = forwardPass(obj, x)
            % keep track of the final activations for each layer
            weightedSums = cell(1, length(obj.Weights));
            activations = cell(1, length(obj.Weights));

            % the first output is the inputs themselves
            weightedSums{1} = x;
            activations{1} = x;

            % iterate over layers (technically, the weights)
            for i = 1:(length(obj.Weights))
                % the weighted sums for the next layer
                % of neurons is just the matrix multiplication
                % of the activations and the weights
                net = activations{i} * obj.Weights{i};
                weightedSums{i + 1} = net;

                % the activation function is then applied
                % to each value (i.e. node) in the resulting matrix
                net = activation(obj, net);
                activations{i + 1} = net;
            end

        end

        %%%%%%%%%%%%%%%%%%%
        % backpropagation %
        %%%%%%%%%%%%%%%%%%%
        function backpropagation(obj, t, weightedSums, activations)
            % compute the cost
            cost = activations{end} - t;

            % deltas go from last layer to first layer
            % but they appear in the list left to right
            % a little confusing
            deltas = cell(1, length(obj.Weights));

            % omfg the chain rule ->

            % the first set of deltas is the last layer
            % its computation is slightly different
            deltas{1} = cost * activationDerivative(obj, weightedSums{end});

            % as we iterate down the weights
            % we also need to iterate up the deltas
            j = 1;

            % the subsequent deltas depend on the most
            % recently computed delta, i.e the backwards
            % propagation
            for i = (length(obj.Weights)):-1:2
                delta = ...
                    (deltas{j} * obj.Weights{i}') ...
                    .* activationDerivative(obj, weightedSums{i} ...
                );
                j = j + 1;
                deltas{j} = delta;
            end

            % <- end chain rule

            % we computed the deltas in reverse order
            % we have to flip them
            deltas = flip(deltas);

            % update the weights
            for i = 1:length(obj.Weights)
                obj.Weights{i} = ...
                    obj.Weights{i} + ( ...
                    -obj.LearningRate * ( ...
                    activations{i}' * deltas{i} ...
                ));
            end

        end

        %%%%%%%%%%%
        % predict %
        %%%%%%%%%%%
        function p = predict(obj, x)
            p = [x ones(size(x, 1), 1)];

            for i = 1:length(obj.Weights)
                p = activation(obj, (p * obj.Weights{i}));
            end

        end

    end

end
