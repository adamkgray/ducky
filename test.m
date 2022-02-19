clear
clc

import ducky.*

net = ducky([2 2 1]);

x = [0 0; 0 1; 1 0; 1 1];
t = [0; 1; 1; 0];

errors = net.train(x, t, 20000);

disp(errors(end))
