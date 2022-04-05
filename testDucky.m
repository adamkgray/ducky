import ducky.*

% data is processed by row (not column as is common in Matlab)
x = [0 0; 0 1; 0 1; 1 1];
t = [0; 1; 1; 0];

% create a 2-2-1 net with alpha=0.1 and a sigmoid activation function
net = ducky([2 2 1], 0.1, "sig");

% train for 1000 epochs
[net, errors] = net.train(x, t, 1000);

plot(errors)
