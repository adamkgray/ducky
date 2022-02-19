clear
clc

import ducky.*

% create two nets to compare
sigNet = ducky([2 2 1], 0.2, 'sig');
tanhNet = ducky([2 2 1], 0.2, 'tanh');

% dataset: XOR
x = [0 0; 0 1; 1 0; 1 1];
t = [0; 1; 1; 0];

% enough to get some results but not too long
epochs = 1000;

% train networks
sigErrors = sigNet.train(x, t, epochs);
tanhErrors = tanhNet.train(x, t, epochs);

% compare error rates over time
plot(sigErrors)
hold on
plot(tanhErrors)
legend('sigmoid', 'tanh')
