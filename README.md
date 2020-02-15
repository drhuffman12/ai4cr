# ai4cr

CircleCI status: [![CircleCI](https://circleci.com/gh/drhuffman12/ai4cr.svg?style=svg)](https://circleci.com/gh/drhuffman12/ai4cr)

[![GitHub release](https://img.shields.io/github/release/drhuffman12/ai4cr.svg)](https://GitHub.com/drhuffman12/ai4cr/releases/)

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

NOTE: `marshal_dump` and `marshal_load` are deprecated; use `to_json` and `from_json` instead, e.g.:

```
# Create and save a net
net = Ai4cr::NeuralNetwork::Backpropagation.new(...)
File.write("../ai4cr_ui/db/seeds/BackpropagationNet.new.json",net.to_json)

# Train and save a net
net.train(some_input, expected_output)
File.write("../ai4cr_ui/db/seeds/BackpropagationNet.trained.json",net.to_json)

# Verify serialization in a spec
json = net.to_json
net2 = Ai4cr::NeuralNetwork::Backpropagation.from_json(json)
assert_approximate_equality_of_nested_list net.weights, net2.weights, 0.000000001
```

## Roadmap

-[ ] Add RNN

-[ ] Port from `ai4r`:
  -[ ] classifiers
    -[ ] classifier.rb
    -[ ] hyperpipes.rb
    -[ ] ib1.rb
    -[ ] id3.rb
    -[ ] multilayer_perceptron.rb
    -[ ] naive_bayes.rb
    -[ ] one_r.rb
    -[ ] prism.rb
    -[ ] simple_linear_regression.rb
    -[ ] votes.rb
      -[ ] zero_r.rb
  -[ ] clusterers
    -[ ] average_linkage.rb
    -[ ] bisecting_k_means.rb
    -[ ] centroid_linkage.rb
    -[ ] clusterer.rb
    -[ ] complete_linkage.rb
    -[ ] diana.rb
    -[ ] k_means.rb
    -[ ] median_linkage.rb
    -[ ] single_linkage.rb
    -[ ] ward_linkage_hierarchical.rb
    -[ ] ward_linkage.rb
      -[ ] weighted_average_linkage.rb
  -[ ] data
    -[ ] data_set.rb
    -[ ] parameterizable.rb
    -[ ] proximity.rb
      -[ ] statistics.rb
  -[ ] experiment
      -[ ] classifier_evaluator.rb
  -[ ] genetic_algorithm
      -[ ] genetic_algorithm.rb
  -[ ] neural_network
    -[x] backpropagation.rb
    -[ ] hopfield.rb
  -[ ] som
    -[ ] layer.rb
    -[ ] node.rb
    -[ ] som.rb
    -[ ] two_phase_layer.rb

If you'd like another class of Ai4r ported, feel free to submit a [new issue](https://github.com/drhuffman12/ai4cr/issues/new).

## Contributing

1. Fork it ( https://github.com/drhuffman12/ai4cr/fork )
2. Create your feature branch (git checkout -b my-new-feature)
3. Commit your changes (git commit -am 'Add some feature')
4. Push to the branch (git push origin my-new-feature)
5. Create a new Pull Request

### (Re-)Format

```bash
docker-compose run app scripts/reformat
```

### Build

```bash
# for a cleaner build:
docker-compose build --force-rm --no-cache --pull

# normally:
docker-compose build
```

### Show version

```bash
docker-compose run app scripts/version_info
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

For any tests that should NEVER fail (e.g.: in spite of sufficient training), put them into `spec`, and run them via:

```bash
$ docker-compose run app scripts/test_always
..............................

Finished in 4.01 milliseconds
30 examples, 0 failures, 0 errors, 0 pending
Execute: 00:00:00.010855717
```

### These will probably SOMETIMES fail!

For any tests that could fails sometimes (e.g.: if not trained enough), put them into `spec_examples`, and run them via:

```bash
$ docker-compose run app scripts/test_sometimes
.............

Finished in 6.76 seconds
16 examples, 0 failures, 0 errors, 0 pending
Execute: 00:00:06.769663359
```

NOTE: That time, it took less than a second to build. I did notice that it took about 10 seconds to build the first run and only less than a second each successive run.
