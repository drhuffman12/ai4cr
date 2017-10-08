# ai4cr

Artificial Intelligence for Crystal (based on https://github.com/SergioFierens/ai4r)

## Installation

Add this to your application's `shard.yml`:

```yaml
dependencies:
  ai4cr:
    github: drhuffman12/ai4cr
```

## Usage

```crystal
require "ai4cr"
```

So far, only Ai4cr::NeuralNetwork::Backpropagation and related tests have been ported.

If you'd like another class of Ai4r ported, feel free to submit a [new issue](https://github.com/drhuffman12/ai4cr/issues/new).

## Development

See docs at: https://drhuffman12.github.io/ai4cr/

See the specs and https://github.com/SergioFierens/ai4r for more info.

## Contributing

1. Fork it ( https://github.com/drhuffman12/ai4cr/fork )
2. Create your feature branch (git checkout -b my-new-feature)
3. Commit your changes (git commit -am 'Add some feature')
4. Push to the branch (git push origin my-new-feature)
5. Create a new Pull Request

## Contributors

- [drhuffman12](https://github.com/drhuffman12) Daniel Huffman - creator, maintainer

## Testing

### Example Spec Results (running on a Lenovo Ideapad y700 w/ i7-6700HQ)

I included the triangle-square-cross training example as test cases.

As expected, the net sometimes correctly guesses all of the examples, more or less, depending on how many times it is trained and various other (random) factors.

Below is an example of the net successfully recognizing all nine test cases.

```bash
$ cd $MY_DEV_FOLDER

$ git clone https://github.com/drhuffman12/ai4cr.git

$ cd ai4cr

$ crystal -v
Crystal 0.23.1 [e2a1389] (2017-07-13) LLVM 3.8.1

$ crystal deps

$ time crystal spec --release --no-debug --time --verbose # These should NEVER fail!
Ai4cr::NeuralNetwork::Backpropagation
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
    when given a net with structure of [3, 2]
      returns an error of type Float64

Finished in 634 microseconds
22 examples, 0 failures, 0 errors, 0 pending
Execute: 00:00:00.0049940

real    0m0.646s
user    0m0.644s
sys     0m0.088s

$ time crystal spec --release --no-debug --time --verbose spec_examples/ # These will probably SOMETIMES fail!
Ai4cr::NeuralNetwork::Backpropagation
  #train
    using image data (input) and shape flags (output) for triangle, square, and cross
      and training 101 times each at a learning rate of 0.697045
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

Finished in 6.37 milliseconds
13 examples, 0 failures, 0 errors, 0 pending
Execute: 00:00:00.0107670

real    0m0.617s
user    0m0.680s
sys     0m0.060s
```

NOTE: That time, it took less than a second to build. I did notice that it took about 10 seconds to build the first run and only less than a second each successive run.
