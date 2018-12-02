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

## Contributing

1. Fork it ( https://github.com/drhuffman12/ai4cr/fork )
2. Create your feature branch (git checkout -b my-new-feature)
3. Commit your changes (git commit -am 'Add some feature')
4. Push to the branch (git push origin my-new-feature)
5. Create a new Pull Request

### Build

```bash
docker-compose build
```

### Show version

```bash
docker-compose run app crystal eval 'require "./src/ai4cr"; puts "Ai4cr version: #{Ai4cr::VERSION}"'
```

### Test

```bash
docker-compose run app crystal spec
```

### Docker console

```bash
docker-compose run app /bin/bash
```

### See also

* The docs at: https://drhuffman12.github.io/ai4cr/

* The specs and https://github.com/SergioFierens/ai4r for more info.

## Contributors

- [drhuffman12](https://github.com/drhuffman12) Daniel Huffman - creator, maintainer

## Example Spec Results (running on a Lenovo Ideapad y700 w/ i7-6700HQ)

I included the triangle-square-cross training example as test cases.

As expected, the net sometimes correctly guesses all of the examples, more or less, depending on how many times it is trained and various other (random) factors.

Below is an example of the net successfully recognizing all nine test cases.

### Setup

```bash
$ cd $MY_DEV_FOLDER

$ git clone https://github.com/drhuffman12/ai4cr.git

$ cd ai4cr

$ docker-compose build
```

### These should NEVER fail!

```bash
$ docker-compose run app crystal spec --release --no-debug --time --error-trace --no-color
............................

Finished in 184 microseconds
28 examples, 0 failures, 0 errors, 0 pending
Execute: 00:00:00.005837007
```

### These will probably SOMETIMES fail!

```bash
$ docker-compose run app crystal spec --release --no-debug --time --error-trace --no-color spec_examples
.............

Finished in 4.1 seconds
13 examples, 0 failures, 0 errors, 0 pending
Execute: 00:00:04.107645264
```

NOTE: That time, it took less than a second to build. I did notice that it took about 10 seconds to build the first run and only less than a second each successive run.


### RNN WIP files

```
crystal spec spec/ai4cr/neural_network/rnn
```

```
crystal spec_examples/ai4cr/neural_network/rnn/tmp_rnn.cr > tmp/tmp_rnn.out
```

```
crystal tool hierarchy spec_examples/ai4cr/neural_network/rnn/tmp_rnn.cr > tmp/tmp_rnn.tool.hierarchy.log
```
