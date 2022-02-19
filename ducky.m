classdef ducky < handle
    % custom feedforward neural net

    properties
        Layers
        LearningRate = 0.1
        Weights
    end

    methods

        %%%%%%%%%%%%%%%
        % constructor %
        %%%%%%%%%%%%%%%

        function obj = ducky(layers)
            % initialise layers
            obj.Layers = layers;

            % initialise weights
            weights = cell(1, length(layers) - 1);

            for i = 1:(length(layers) - 1)
                n = layers(i) + 1;
                m = layers(i + 1) + 1;

                % the last weight layer is different
                if i == (length(layers) - 1)
                    m = layers(end);
                end

                w = normrnd(0, 1, [n m]) / sqrt(layers(i));
                weights{i} = w;
            end

            obj.Weights = weights;
        end

        %%%%%%%%%%%%%%%%%%%%%%%
        % activation function %
        %%%%%%%%%%%%%%%%%%%%%%%
        function y = activation(~, x)
            y = 1 ./ (1 + exp(-x));
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % derivative of the activation function %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function y = activationDerivative(~, x)
            f = 1 ./ (1 + exp(-x));
            y = f .* (1 - f);
        end

        %%%%%%%%%%%%%%%%%%%%%
        % sum squared error %
        %%%%%%%%%%%%%%%%%%%%%
        function errors = sse(obj, x, t)
            p = predict(obj, x);
            errors = sum((t - p).^2) / 2;
        end

        %%%%%%%%%%%%%%%%%
        % train network %
        %%%%%%%%%%%%%%%%%
        function errors = train(obj, x, t, epochs)
            % add bias feature
            xWithBias = [x ones(size(x, 1), 1)];

            % keep track of SSE
            errors = zeros(1, epochs);

            for e = 1:epochs

                % fit training data
                for i = 1:size(xWithBias, 1)
                    fit(obj, xWithBias(i, :), t(i, :))
                end

                % remember SSE for this epoch
                errors(e) = sse(obj, x, t);
            end

        end

        %%%%%%%
        % fit %
        %%%%%%%
        function fit(obj, x, t)

            %%%%%%%%%%%%%%%%
            % forward pass %
            %%%%%%%%%%%%%%%%

            % keep track of the final activations for each layer
            weightedSums = cell(1, length(obj.Weights));
            activations = cell(1, length(obj.Weights));

            % the first output is the inputs themselves
            weightedSums{1} = x;
            activations{1} = x;

            % iterate over layers (technically, the weights)
            for i = 1:(length(obj.Weights))
                net = activations{i} * obj.Weights{i};
                weightedSums{i + 1} = net;

                net = activation(obj, net);
                activations{i + 1} = net;
            end

            %%%%%%%%%%%%%%%%%%%
            % backpropagation %
            %%%%%%%%%%%%%%%%%%%

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
                delta = (deltas{j} * obj.Weights{i}') .* activationDerivative(obj, weightedSums{i});
                j = j + 1;
                deltas{j} = delta;
            end

            % <- end chain rule

            % we computed the deltas in reverse order
            % we have to flip them
            deltas = flip(deltas);

            % update the weights
            for i = 1:length(obj.Weights)
                obj.Weights{i} = obj.Weights{i} + (-obj.LearningRate * (activations{i}' * deltas{i}));
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
