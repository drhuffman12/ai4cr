# Example Spec Results (running on a Lenovo Ideapad y700 w/ i7-6700HQ)

I included the triangle-square-cross training example as test cases.

As expected, the net sometimes correctly guesses all of the examples, more or less, depending on how many times it is trained and various other (random) factors.

Below is an example of the net successfully recognizing all nine test cases.

```

$ crystal -v
Crystal 0.23.1 [e2a1389] (2017-07-13) LLVM 3.8.1

$ time crystal spec --release --no-debug --time --verbose
Ai4c::NeuralNetwork::Backpropagation
  #init_network
    when given a net with structure of [4, 2]
      sets @activation_nodes to expected nested array
      sets @weights to expected size
      sets @weights.first to expected size
      sets each sub-array w/in @weights.first to expected size
    when given a net with structure of [2, 2, 1]
      sets @activation_nodes to expected nested array
      sets @weights to expected size
      sets @weights.first to expected size
      sets each sub-array w/in @weights.first to expected size
    when given a net with structure of [2, 2, 1] with bias disabled
      sets @activation_nodes to expected nested array
      sets @weights to expected size
      sets @weights.first to expected size
      sets each sub-array w/in @weights.first to expected size
  #eval
    when given a net with structure of [3, 2]
      returns output nodes of expected size
    when given a net with structure of [2, 4, 8, 10, 7]
      returns output nodes of expected size
  #dump
    when given a net with structure of [3, 2]
      @structure of the dumped net matches @structure of the loaded net
      @disable_bias on the dumped net matches @disable_bias of the loaded net
      @learning_rate of the dumped net approximately matches @learning_rate of the loaded net
      @momentum of the dumped net approximately matches @momentum of the loaded net
      @weights of the dumped net approximately matches @weights of the loaded net
      @last_changes of the dumped net approximately matches @last_changes of the loaded net
      @activation_nodes of the dumped net approximately matches @activation_nodes of the loaded net
  #train
    using image data (input) and shape flags (output) for triangle, square, and cross
      and training 96 times each at a learning rate of 0.101981
        error_averages
          decrease (i.e.: first > last)
          should end up close to 0.1 +/- 0.1
          should end up close to 0.01 +/- 0.01
          should end up close to 0.001 +/- 0.001
        #eval correctly guesses shape flags (output) when given image data (input) of
          original input data for
            TRIANGLE
            SQUARE
            CROSS
          noisy input data for
            TRIANGLE
            SQUARE
            CROSS
          base noisy input data for
            TRIANGLE
            SQUARE
            CROSS

Finished in 22.5 milliseconds
34 examples, 0 failures, 0 errors, 0 pending
Execute: 00:00:00.0353190

real    0m2.160s
user    0m2.304s
sys     0m0.236s
```

NOTE: That time, it took about a couple seconds to build. I did notice that it took about 40 seconds to build the first run and only a couple each successive run.