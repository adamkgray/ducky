import ducky.*

% data is processed by row (not column as is common in Matlab)
x = [0 0; 0 1; 1 0; 1 1];
t = [0; 1; 1; 0];

% create a 2-3-1 net with alpha=0.1 and a tanh activation function
net = ducky([2 3 1], 0.1, "tanh");

% train for 1000 epochs
[net, errors] = net.train(x, t, 1000);

% plot error rate over time
figure(1)
plot(errors)
title("error rate over epochs")

% display final weights
figure(2)
net.weightsHeatMap

% display learned classification space
figure(3)
x = 0:0.05:1;
x = nchoosek(x, 2);
x = [x; fliplr(x)];
y = net.predict(x);
plot3(x(:, 1), x(:, 2), y, 'o');
title("classification space")
