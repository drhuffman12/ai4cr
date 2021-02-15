require "./../../../spec_helper"
require "./../../../spectator_helper"

def puts_debug(message = "")
  puts message if ENV.has_key?("DEBUG") && ENV["DEBUG"] == "1"
end

Spectator.describe Ai4cr::NeuralNetwork::Cmn::MiniNetManager do
  let(my_breed_manager) { Ai4cr::NeuralNetwork::Cmn::MiniNetManager.new }

  before_each do
    my_breed_manager.counter.reset!
  end

  describe "#to_json" do
    context "for a new MiniNetManager" do
      context "correctly exports" do
        it "the whole initial object" do
          # my_breed_manager.counter.inc("foo")
          counter = my_breed_manager.counter
          puts_debug
          puts_debug "counter.to_json: #{counter.to_json}"
          puts_debug

          # NOTE: 'exported' vs 'expected'
          exported_json = my_breed_manager.to_json
          expected_json = "{}"

          expect(exported_json).to be_a(String)
          expect(exported_json).to eq(expected_json)
        end
      end
    end
  end

  describe "#to_json and #from_json" do
    context "for a new MiniNetManager" do
      context "correctly exports and imports" do
        it "the whole object" do
          exported_json = my_breed_manager.to_json
          imported = my_breed_manager.class.from_json(exported_json)
          re_exported_json = imported.to_json

          expect(re_exported_json).to eq(exported_json)
        end
      end
    end
  end

  describe "For Adam and Eve examples" do
    let(inputs) { [0.1, 0.2] }
    let(outputs) { [0.3, 0.4] }

    # TODO: Split this up into smaller tests!
    let(delta_child_1) { (rand*2 - 0.5) }
    let(delta_child_2) { (rand*2 - 0.5) }

    let(ancestor_adam_value) { 0.1 }
    let(ancestor_eve_value) { 0.9 }
    let(expected_child_1_value) { ancestor_adam_value + delta_child_1 * (ancestor_eve_value - ancestor_adam_value) }

    let(ancestor_adam) {
      ancestor = my_breed_manager.create(
        name: "Adam",
        bias_default: ancestor_adam_value / 2.0,
        learning_rate: ancestor_adam_value,
        momentum: 1.0 - ancestor_adam_value,
        deriv_scale: ancestor_adam_value / 4.0
      )
      ancestor.weights = [
        [-0.1, 0.2],
        [-0.3, 0.4],
        [-0.5, 0.6],
      ]
      ancestor.train(inputs, outputs)
      ancestor
    }
    let(ancestor_eve) {
      ancestor = my_breed_manager.create(
        name: "Eve",
        bias_default: ancestor_eve_value,
        learning_rate: ancestor_eve_value,
        momentum: 1.0 - ancestor_eve_value,
        deriv_scale: ancestor_eve_value / 4.0
      )
      ancestor.weights = [
        [0.1, -0.2],
        [0.3, -0.4],
        [0.5, -0.6],
      ]
      ancestor.train(inputs, outputs)
      ancestor
    }

    it "birth_id's are in the consistent order (when birthed in order)" do
      expected_birth_counter = 0
      puts_debug
      puts_debug "ancestor_adam: #{ancestor_adam.to_json}"
      puts_debug
      puts_debug "ancestor_eve: #{ancestor_eve.to_json}"

      expected_birth_counter += 1

      expect(ancestor_adam.birth_id).to eq(expected_birth_counter)

      expected_birth_counter += 1
      expect(ancestor_eve.birth_id).to eq(expected_birth_counter)

      # cain
      child_1 = my_breed_manager.breed(ancestor_adam, ancestor_eve, delta: delta_child_1)
      child_1.name = "Cain, child of #{ancestor_adam.name} and #{ancestor_eve.name}"

      puts_debug "child_1: #{child_1.to_json}"
      expected_birth_counter += 1
      expect(child_1.birth_id).to eq(expected_birth_counter)

      # abel
      child_2 = my_breed_manager.breed(ancestor_adam, ancestor_eve, delta: delta_child_2)
      child_2.name = "Abel, child of #{child_2.name} and #{ancestor_eve.name}"

      puts_debug "child_2: #{child_2.to_json}"
      expected_birth_counter += 1
      expect(child_2.birth_id).to eq(expected_birth_counter)

      puts_debug
      puts_debug "Now, in order or youngest to oldest:"
      [ancestor_adam, ancestor_eve, child_1, child_2].sort_by do |person|
        (-person.birth_id)
      end.each do |person|
        puts_debug
        puts_debug "person: #{person.to_json}"
      end
    end

    describe "#breed" do
      let(ancestor_1) { my_breed_manager.create(name: "defaults 1") }
      let(ancestor_2) { my_breed_manager.create(name: "defaults 2") }
      let(ancestor_3) {
        my_breed_manager.create(name: "non-default width", width: 4)
      }
      let(ancestor_4) {
        my_breed_manager.create(name: "non-default height", height: 5)
      }
      let(ancestor_5) {
        my_breed_manager.create(name: "non-default width and height", width: 4, height: 5)
      }
      context "when parents have same width and height" do
        it "does NOT raise" do
          ancestor_a = ancestor_1
          ancestor_b = ancestor_2

          expect(ancestor_a.width).to eq(ancestor_b.width)
          expect(ancestor_a.height).to eq(ancestor_b.height)

          expect { my_breed_manager.breed(ancestor_a, ancestor_b) }.not_to raise_error
        end
      end
      context "when parents have differing width" do
        it "raises" do
          ancestor_a = ancestor_1
          ancestor_b = ancestor_3

          expect(ancestor_a.width).not_to eq(ancestor_b.width)
          expect(ancestor_a.height).to eq(ancestor_b.height)

          expect { my_breed_manager.breed(ancestor_a, ancestor_b) }.to raise_error("Parents must be the same width and height")
        end
      end
      context "when parents have differing height" do
        it "raises" do
          ancestor_a = ancestor_1
          ancestor_b = ancestor_4

          expect(ancestor_a.width).to eq(ancestor_b.width)
          expect(ancestor_a.height).not_to eq(ancestor_b.height)

          expect { my_breed_manager.breed(ancestor_a, ancestor_b) }.to raise_error("Parents must be the same width and height")
        end
      end
      context "when parents have differing width and height" do
        it "raises" do
          ancestor_a = ancestor_1
          ancestor_b = ancestor_5

          expect(ancestor_a.width).not_to eq(ancestor_b.width)
          expect(ancestor_a.height).not_to eq(ancestor_b.height)

          expect { my_breed_manager.breed(ancestor_a, ancestor_b) }.to raise_error("Parents must be the same width and height")
        end
      end
    end

    context "children have expected values for" do
      let(child_1) {
        # cain
        child = my_breed_manager.breed(ancestor_adam, ancestor_eve, delta: delta_child_1)
        child.name = "Cain, child of #{ancestor_adam.name} and #{ancestor_eve.name}"
        child
      }

      let(ancestor_adam_json) { JSON.parse(ancestor_adam.to_json) }
      let(ancestor_eve_json) { JSON.parse(ancestor_eve.to_json) }
      let(child_1_json) { JSON.parse(child_1.to_json) }

      describe "#to_json" do
        context "for a new MiniNetManager with two initial 'ancestors' and one 'child'" do
          context "correctly exports" do
            it "the whole initial object" do
              expect(ancestor_adam).not_to be_nil
              expect(ancestor_eve).not_to be_nil
              expect(child_1).not_to be_nil

              # my_breed_manager.counter.inc("foo")
              counter = my_breed_manager.counter
              puts_debug
              puts_debug "counter.to_json: #{counter.to_json}"
              puts_debug
              puts_debug "my_breed_manager.to_json: #{my_breed_manager.to_json}"
              puts_debug

              # NOTE: 'exported' vs 'expected'
              exported_json = my_breed_manager.to_json
              expected_json = "{}"

              expect(exported_json).to be_a(String)
              expect(exported_json).to eq(expected_json)
            end
          end
        end
      end

      context "#parts_to_copy stay the same, such as" do
        it "misc instance variables" do
          puts_debug
          puts_debug "  ancestor_adam.to_json: #{ancestor_adam.to_json}"
          puts_debug "  ancestor_eve.to_json: #{ancestor_eve.to_json}"
          puts_debug "  child_1.to_json: #{child_1.to_json}"
          puts_debug

          [
            "width", "height", "height_considering_bias", "width_indexes", "height_indexes",
            "learning_style", "disable_bias", "outputs_expected",
          ].each do |var|
            puts_debug
            puts_debug "var: #{var}"
            puts_debug
            expect(ancestor_adam_json[var]).to eq(ancestor_eve_json[var])
            expect(child_1_json[var]).to eq(ancestor_adam_json[var])
          end
        end
      end

      context "#mix_parts are delta-mixed, such as" do
        it "bias_default" do
          ancestor_adam_value = ancestor_adam.bias_default
          ancestor_eve_value = ancestor_eve.bias_default
          expected_child_1_value = my_breed_manager.mix_nested_parts(ancestor_adam_value, ancestor_eve_value, delta_child_1)

          expect(ancestor_adam.bias_default).not_to eq(ancestor_eve.bias_default)
          expect(child_1.bias_default).to eq(expected_child_1_value)
        end

        it "learning_rate" do
          ancestor_adam_value = ancestor_adam.learning_rate
          ancestor_eve_value = ancestor_eve.learning_rate
          expected_child_1_value = my_breed_manager.mix_nested_parts(ancestor_adam_value, ancestor_eve_value, delta_child_1)

          expect(ancestor_adam.learning_rate).not_to eq(ancestor_eve.learning_rate)
          expect(child_1.learning_rate).to eq(expected_child_1_value)
        end

        it "momentum" do
          ancestor_adam_value = ancestor_adam.momentum
          ancestor_eve_value = ancestor_eve.momentum
          expected_child_1_value = my_breed_manager.mix_nested_parts(ancestor_adam_value, ancestor_eve_value, delta_child_1)

          expect(ancestor_adam.momentum).not_to eq(ancestor_eve.momentum)
          expect(child_1.momentum).to eq(expected_child_1_value)
        end

        it "deriv_scale" do
          ancestor_adam_value = ancestor_adam.deriv_scale
          ancestor_eve_value = ancestor_eve.deriv_scale
          expected_child_1_value = my_breed_manager.mix_nested_parts(ancestor_adam_value, ancestor_eve_value, delta_child_1)

          expect(ancestor_adam.deriv_scale).not_to eq(ancestor_eve.deriv_scale)
          expect(child_1.deriv_scale).to eq(expected_child_1_value)
        end

        it "weights" do
          ancestor_adam_value = ancestor_adam.weights
          ancestor_eve_value = ancestor_eve.weights
          expected_child_1_value = my_breed_manager.mix_nested_parts(ancestor_adam_value, ancestor_eve_value, delta_child_1)

          expect(ancestor_adam.weights).not_to eq(ancestor_eve.weights)
          expect(child_1.weights).to eq(expected_child_1_value)
        end

        it "inputs_given" do
          ancestor_adam_value = ancestor_adam.inputs_given
          ancestor_eve_value = ancestor_eve.inputs_given
          expected_child_1_value = my_breed_manager.mix_nested_parts(ancestor_adam_value, ancestor_eve_value, delta_child_1)

          expect(ancestor_adam.inputs_given).not_to eq(ancestor_eve.inputs_given)
          expect(child_1.inputs_given).to eq(expected_child_1_value)
        end

        it "outputs_guessed" do
          ancestor_adam_value = ancestor_adam.outputs_guessed
          ancestor_eve_value = ancestor_eve.outputs_guessed

          expected_child_1_value = my_breed_manager.mix_nested_parts(ancestor_adam_value, ancestor_eve_value, delta_child_1)

          expect(ancestor_adam.outputs_guessed).not_to eq(ancestor_eve.outputs_guessed)
          expect(child_1.outputs_guessed).to eq(expected_child_1_value)
        end

        it "output_deltas" do
          ancestor_adam_value = ancestor_adam.output_deltas
          ancestor_eve_value = ancestor_eve.output_deltas
          expected_child_1_value = my_breed_manager.mix_nested_parts(ancestor_adam_value, ancestor_eve_value, delta_child_1)

          expect(ancestor_adam.output_deltas).not_to eq(ancestor_eve.output_deltas)
          expect(child_1.output_deltas).to eq(expected_child_1_value)
        end

        it "last_changes" do
          ancestor_adam_value = ancestor_adam.last_changes
          ancestor_eve_value = ancestor_eve.last_changes
          expected_child_1_value = my_breed_manager.mix_nested_parts(ancestor_adam_value, ancestor_eve_value, delta_child_1)

          expect(ancestor_adam.last_changes).not_to eq(ancestor_eve.last_changes)
          expect(child_1.last_changes).to eq(expected_child_1_value)
        end

        it "output_errors" do
          ancestor_adam_value = ancestor_adam.output_errors
          ancestor_eve_value = ancestor_eve.output_errors
          expected_child_1_value = my_breed_manager.mix_nested_parts(ancestor_adam_value, ancestor_eve_value, delta_child_1)

          expect(ancestor_adam.output_errors).not_to eq(ancestor_eve.output_errors)
          expect(child_1.output_errors).to eq(expected_child_1_value)
        end

        it "input_deltas" do
          ancestor_adam_value = ancestor_adam.input_deltas
          ancestor_eve_value = ancestor_eve.input_deltas
          expected_child_1_value = my_breed_manager.mix_nested_parts(ancestor_adam_value, ancestor_eve_value, delta_child_1)

          expect(ancestor_adam.input_deltas).not_to eq(ancestor_eve.input_deltas)
          expect(child_1.input_deltas).to eq(expected_child_1_value)
        end
      end

      context "NOT copied and NOT mixed, but instead freshly initialized" do
        it "foo debug" do
          puts_debug
          puts_debug "ancestor_adam: #{ancestor_adam.to_json}"
          puts_debug
          puts_debug "ancestor_eve: #{ancestor_eve.to_json}"
          puts_debug
          puts_debug "child_1: #{child_1.to_json}"
          puts_debug
        end

        context "error_stats" do
          it "history_size" do
            # TODO: Adjust to compare error_stats
            ancestor_adam_value = ancestor_adam.error_stats.history_size
            ancestor_eve_value = ancestor_eve.error_stats.history_size
            # expected_child_1_value = my_breed_manager.mix_nested_parts(ancestor_adam_value, ancestor_eve_value, delta_child_1)
            expected_child_1_value = ancestor_adam_value

            expect(ancestor_adam_value).to eq(ancestor_eve_value)
            expect(child_1.error_stats.history_size).to eq(expected_child_1_value)
          end

          it "distance" do
            # TODO: Adjust to compare error_stats
            ancestor_adam_value = ancestor_adam.error_stats.distance
            ancestor_eve_value = ancestor_eve.error_stats.distance
            # expected_child_1_value = my_breed_manager.mix_nested_parts(ancestor_adam_value, ancestor_eve_value, delta_child_1)
            expected_child_1_value = -1.0

            expect(ancestor_adam_value).not_to eq(ancestor_eve_value)
            expect(child_1.error_stats.distance).to eq(expected_child_1_value)
          end

          it "history.size" do
            # TODO: Adjust to compare error_stats
            ancestor_adam_value = ancestor_adam.error_stats.history.size
            ancestor_eve_value = ancestor_eve.error_stats.history.size
            # expected_child_1_value = my_breed_manager.mix_nested_parts(ancestor_adam_value, ancestor_eve_value, delta_child_1)
            expected_child_1_value = 0

            expect(ancestor_adam_value).to eq(ancestor_eve_value)
            expect(child_1.error_stats.history.size).to eq(expected_child_1_value)
          end

          it "score" do
            # TODO: Adjust to compare error_stats
            ancestor_adam_value = ancestor_adam.error_stats.score
            ancestor_eve_value = ancestor_eve.error_stats.score
            # expected_child_1_value = my_breed_manager.mix_nested_parts(ancestor_adam_value, ancestor_eve_value, delta_child_1)
            expected_child_1_value = 1.8446744073709552e+19

            expect(ancestor_adam_value).not_to eq(ancestor_eve_value)
            expect(child_1.error_stats.score).to eq(expected_child_1_value)
          end
        end
      end
    end
  end
end
