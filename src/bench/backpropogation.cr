require "../ai4cr"

shape = [256, 3]
# shape = [256, 1000, 3]
# shape = [256, 500, 500, 3]
Ai4cr::NeuralNetwork::Backpropagation.new(shape)

# USAGE:
# crystal build --release src/bench/backpropogation.cr -o bin/bench/backpropogation
# time bin/bench/backpropogation

# mkdir -p tmp/bench

# valgrind --tool=callgrind --inclusive=yes --tree=both --auto=yes --cache-sim=yes --branch-sim=yes --callgrind-out-file=tmp/bench/backpropogation.callgrind.out bin/bench/backpropogation

# valgrind --tool=callgrind --cache-sim=yes --branch-sim=yes --callgrind-out-file=tmp/bench/backpropogation.callgrind.out bin/bench/backpropogation


