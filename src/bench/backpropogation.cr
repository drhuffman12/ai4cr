require "../ai4cr"

shape = [256, 3]
# shape = [256, 1000, 3]
# shape = [256, 500, 500, 3]
Ai4cr::NeuralNetwork::Backpropagation::Net.new(shape)

# USAGE:
# crystal build --release src/bench/backpropogation2.cr -o bin/bench/backpropogation2
# crystal build --release --debug src/bench/backpropogation2.cr -o bin/bench/backpropogation2
# crystal build src/bench/backpropogation2.cr -o bin/bench/backpropogation2
# time bin/bench/backpropogation2

# mkdir -p tmp/bench

# valgrind --tool=callgrind --inclusive=yes --tree=both --auto=yes --cache-sim=yes --branch-sim=yes --callgrind-out-file=tmp/bench/backpropogation2.callgrind.out bin/bench/backpropogation2

# valgrind --tool=callgrind --cache-sim=yes --branch-sim=yes --callgrind-out-file=tmp/bench/backpropogation2.callgrind.out bin/bench/backpropogation2

# valgrind --tool=callgrind --cache-sim=yes --branch-sim=yes --dump-instr=yes --collect-jumps=yes --callgrind-out-file=tmp/bench/backpropogation2.callgrind.out bin/bench/backpropogation2
