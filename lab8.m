 %%%
clear;
close all;

%%% activation function
sigmoid = @(s) 1 ./ (1 + exp(-s));
dsigmoid = @(s) sigmoid(s) .* (1 - sigmoid(s));
relu = @(s) max(0, s);
drelu = @(s) s > 0;

fs = {[], relu, relu, relu, relu, relu, relu, sigmoid};
dfs = {[], drelu, drelu, drelu, drelu, drelu, drelu, dsigmoid};

%%% load data
load dat.mat
volume = 1024;

%%% choose parameter
alpha = 0.01;
epochs = 500;
J = zeros(1, epochs);
mini_batch = 100;
Acc = zeros(1, trainSize / mini_batch);

%%% define network structure
structure = [1024 0
             0 64
             0 64
             0 32
             0 32
             0 16
             0 16
             0 1];
L = size(structure, 1);
W = cell(L-1, 1);
Z = cell(L, 1);
A = cell(L, 1);
X = cell(L, 1);
DELTA = cell(L, 1);

%%% initialize weights
for l=1:L-1
    %W{l} = randn(structure(l+1, 2), sum(structure(l, :)));
    W{l} = (rand(structure(l + 1, 2), sum(structure(l, :))) * 2 - 1) * sqrt(6 / (structure(l + 1, 2) + sum(structure(l, :))));
end

%%% training
for t=1:epochs
    ind = randperm(trainSize);
    y = zeros(1, mini_batch);
    for k=1:ceil(trainSize/mini_batch)
        % first layer of internal input : zeroes
        Z{1} = zeros(structure(1,2), mini_batch);
        %A{1} = relu(Z{1});
        A{1} = relu(Z{1});
        % first layer of extenal input
        X{1} = trainSrc(:,ind((k-1) * mini_batch + 1 : min(k * mini_batch, trainSize)));
        % label
        y = trainLabels(:,ind((k-1) * mini_batch + 1 : min(k * mini_batch, trainSize)));
        % forward computing
        for l=1:L-1
            [A{l+1}, Z{l+1}] = fc(W{l}, A{l}, X{l}, fs{l + 1});
        end
        % J(t) : MSE
        %J(t) = 1/2/mini_batch*sum((A{L}(:)-y(:)).^2);
        % J(t) : Binary Cross-Entropy
        all_one = ones(1, mini_batch);
        %temp = -(y .* log(A{L}) + (all_one - y) .* log(all_one - A{L}));
        J(t) = sum(-(y .* log(A{L} + 1e-8) + (all_one - y) .* log(all_one - A{L}) + 1e-8), 2) / mini_batch;
        % backward computation
        % MSE
        %DELTA{L} = (A{L} - y) .* dsigmoid(Z{L});
        % Binary Cross-Entropy
        DELTA{L} = -dsigmoid(Z{L}) .* (y - A{L}) ./ ((A{L} + 1e-8) .* (all_one - A{L}));
        for l=L-1:-1:2
            DELTA{l} = bc(W{l}, Z{l}, DELTA{l+1}, dfs{l + 1});
        end
        % computing gradients and update the weights
        for l=1:L-1
            dW = DELTA{l+1} * [X{l};A{l}]' / mini_batch;
            W{l} = W{l} - alpha * dW;
        end
        % to compute the accuracy during training
        for p=1:size(A{L}, 2)
            if A{L}(p) >= 0.5
                A{L}(p) = 1;
            else
                A{L}(p) = 0;
            end
        end
        count = 0;
        for p=1:size(A{L}, 2)
            if A{L}(p) == y(p)
                count = count + 1;
            end
        end
        Acc(t) = count / mini_batch;
    end
    fprintf('%i/%i epochs: J=%.4f\n', t, epochs, J(t));
end

% save model
save model.mat W structure

% painting
figure
plot(J)
figure
%fprintf("Accuracy on train set is %f%\n", mean(Acc));
plot(Acc)

%%% training
count = 0;
for i=1:size(trainLabels, 2)
    Z{1} = zeros(structure(1,2), 1);
    A{1} = relu(Z{1});
    X{1} = trainSrc(:,i);
    for p=1:L-1
        [A{p+1}, Z{p+1}] = fc(W{p}, A{p}, X{p}, fs{p + 1});
    end
    if A{L} >= 0.5
        A{L} = 1;
    else
        A{L} = 0;
    end
    if A{L} == trainLabels(i)
        count = count + 1;
    end
end
fprintf("Accuracy on train set is %f%\n", count / size(trainLabels, 2));

%%% testing
count = 0;
for i=1:size(testLabels, 2)
    Z{1} = zeros(structure(1,2), 1);
    A{1} = relu(Z{1});
    X{1} = testSrc(:,i);
    for p=1:L-1
        [A{p+1}, Z{p+1}] = fc(W{p}, A{p}, X{p}, fs{p + 1});
    end
    if A{L} >= 0.5
        A{L} = 1;
    else
        A{L} = 0;
    end
    if A{L} == testLabels(i)
        count = count + 1;
    end
end
fprintf("Accuracy on test set is %f%\n", count / size(testLabels, 2));