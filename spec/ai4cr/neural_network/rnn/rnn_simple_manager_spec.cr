require "./../../../spec_helper"
require "./../../../spectator_helper"

Spectator.describe Ai4cr::NeuralNetwork::Rnn::RnnSimpleManager do
  let(my_breed_manager) { Ai4cr::NeuralNetwork::Rnn::RnnSimpleManager.new }

  describe "For Adam and Eve examples" do
    let(delta_child_1) { Ai4cr::Data::Utils.rand_neg_half_to_pos_one_and_half_no_zero_no_one }
    let(delta_child_2) { Ai4cr::Data::Utils.rand_neg_half_to_pos_one_and_half_no_zero_no_one }

    let(ancestor_adam_learning_rate_expected) { 0.1 }
    let(ancestor_eve_learning_rate_expected) { 0.9 }

    let(config_default_randomized) {
      Ai4cr::NeuralNetwork::Rnn::RnnSimple.new.config
    }

    let(config_adam) {
      config_default_randomized.merge(
        name: "Adam", learning_rate: ancestor_adam_learning_rate_expected,
        momentum: Ai4cr::Data::Utils.rand_excluding,
        deriv_scale: Ai4cr::Data::Utils.rand_excluding / 2.0
      )
    }

    let(config_eve) {
      config_default_randomized.merge(
        name: "Eve", learning_rate: ancestor_eve_learning_rate_expected,
        momentum: Ai4cr::Data::Utils.rand_excluding,
        deriv_scale: Ai4cr::Data::Utils.rand_excluding / 2.0
      )
    }

    let(ancestor_adam) { my_breed_manager.create(**config_adam) }
    let(ancestor_eve) { my_breed_manager.create(**config_eve) }

    before_each do
      my_breed_manager.counter_reset
    end

    # describe "#breed" do
    #   let(ancestor_1) { my_breed_manager.create(name: "defaults 1") }
    #   let(ancestor_2) { my_breed_manager.create(name: "defaults 2") }

    #   let(ancestor_3) { my_breed_manager.create(name: "non-default io_offset", io_offset: RnnSimple::IO_OFFSET_DEFAULT + 1 + rand(2)) }
    #   let(ancestor_3) { my_breed_manager.create(name: "non-default time_col_qty", time_col_qty: RnnSimple::TIME_COL_QTY_MIN + 1 + rand(2)) }
    #   let(ancestor_3) { my_breed_manager.create(name: "non-default input_size", input_size: RnnSimple::INPUT_SIZE_MIN + 1 + rand(2)) }
    #   let(ancestor_3) { my_breed_manager.create(name: "non-default output_size", output_size: RnnSimple::OUTPUT_SIZE_MIN + 1 + rand(2)) }
    #   let(ancestor_3) { my_breed_manager.create(name: "non-default hidden_layer_qty", hidden_layer_qty: RnnSimple::HIDDEN_LAYER_QTY_MIN + 1 + rand(2) }
    #   let(ancestor_3) { my_breed_manager.create(name: "non-default hidden_size_given", hidden_size_given: RnnSimple::HIDDEN_SIZE_DEFAULT + 1 + rand(2)) }
    #   let(ancestor_3) { my_breed_manager.create(name: "non-default learning_style", learning_style: RnnSimple::IO_OFFSET_DEFAULT + 1 + rand(2)) }
    #   let(ancestor_3) { my_breed_manager.create(name: "non-default disable_bias", disable_bias: true) }

    #   # let(ancestor_4) {
    #   #   my_breed_manager.create(name: "non-default output_size", output_size: 5)
    #   # }
    #   # let(ancestor_5) {
    #   #   my_breed_manager.create(name: "non-default output_size and input_size", output_size: 4, input_size: 5)
    #   # }
    #   context "when parents have same io_offset and time_col_qty  and output_size and input_size" do
    #     it "does NOT raise" do
    #       ancestor_a = ancestor_1
    #       ancestor_b = ancestor_2

    #       expect(ancestor_a.io_offset).to eq(ancestor_b.io_offset)
    #       expect(ancestor_a.time_col_qty).to eq(ancestor_b.time_col_qty)
    #       expect(ancestor_a.input_size).to eq(ancestor_b.input_size)
    #       expect(ancestor_a.output_size).to eq(ancestor_b.output_size)

    #       expect { my_breed_manager.breed(ancestor_a, ancestor_b) }.not_to raise_error
    #     end
    #   end

    #   context "when parents have differing" do
    #     context "io_offset" do
    #       it "raises" do
    #         ancestor_a = ancestor_1
    #         ancestor_b = ancestor_4

    #         expect(ancestor_a.output_size).to eq(ancestor_b.output_size)
    #         expect(ancestor_a.input_size).not_to eq(ancestor_b.input_size)

    #         expect { my_breed_manager.breed(ancestor_a, ancestor_b) }.to raise_error("Parents must be the same output_size and input_size")
    #       end
    #     end

    #     context "time_col_qty" do
    #       it "raises" do
    #         ancestor_a = ancestor_1
    #         ancestor_b = ancestor_4

    #         expect(ancestor_a.output_size).to eq(ancestor_b.output_size)
    #         expect(ancestor_a.input_size).not_to eq(ancestor_b.input_size)

    #         expect { my_breed_manager.breed(ancestor_a, ancestor_b) }.to raise_error("Parents must be the same output_size and input_size")
    #       end
    #     end

    #     context "input_size" do
    #       it "raises" do
    #         ancestor_a = ancestor_1
    #         ancestor_b = ancestor_4

    #         expect(ancestor_a.output_size).to eq(ancestor_b.output_size)
    #         expect(ancestor_a.input_size).not_to eq(ancestor_b.input_size)

    #         expect { my_breed_manager.breed(ancestor_a, ancestor_b) }.to raise_error("Parents must be the same output_size and input_size")
    #       end
    #     end

    #     context "output_size" do
    #       it "raises" do
    #         ancestor_a = ancestor_1
    #         ancestor_b = ancestor_4

    #         expect(ancestor_a.output_size).to eq(ancestor_b.output_size)
    #         expect(ancestor_a.input_size).not_to eq(ancestor_b.input_size)

    #         expect { my_breed_manager.breed(ancestor_a, ancestor_b) }.to raise_error("Parents must be the same output_size and input_size")
    #       end
    #     end

    #     context "hidden_layer_qty" do
    #       it "raises" do
    #         ancestor_a = ancestor_1
    #         ancestor_b = ancestor_4

    #         expect(ancestor_a.output_size).to eq(ancestor_b.output_size)
    #         expect(ancestor_a.input_size).not_to eq(ancestor_b.input_size)

    #         expect { my_breed_manager.breed(ancestor_a, ancestor_b) }.to raise_error("Parents must be the same output_size and input_size")
    #       end
    #     end

    #     context "hidden_size" do
    #       it "raises" do
    #         ancestor_a = ancestor_1
    #         ancestor_b = ancestor_4

    #         expect(ancestor_a.output_size).to eq(ancestor_b.output_size)
    #         expect(ancestor_a.input_size).not_to eq(ancestor_b.input_size)

    #         expect { my_breed_manager.breed(ancestor_a, ancestor_b) }.to raise_error("Parents must be the same output_size and input_size")
    #       end
    #     end

    #     context "learning_style" do
    #       it "raises" do
    #         ancestor_a = ancestor_1
    #         ancestor_b = ancestor_4

    #         expect(ancestor_a.output_size).to eq(ancestor_b.output_size)
    #         expect(ancestor_a.input_size).not_to eq(ancestor_b.input_size)

    #         expect { my_breed_manager.breed(ancestor_a, ancestor_b) }.to raise_error("Parents must be the same output_size and input_size")
    #       end
    #     end

    #     context "disable_bias" do
    #       it "raises" do
    #         ancestor_a = ancestor_1
    #         ancestor_b = ancestor_4

    #         expect(ancestor_a.output_size).to eq(ancestor_b.output_size)
    #         expect(ancestor_a.input_size).not_to eq(ancestor_b.input_size)

    #         expect { my_breed_manager.breed(ancestor_a, ancestor_b) }.to raise_error("Parents must be the same output_size and input_size")
    #       end
    #     end
    #   end

    #   context "when parents have differing output_size" do
    #     it "raises" do
    #       ancestor_a = ancestor_1
    #       ancestor_b = ancestor_3

    #       expect(ancestor_a.output_size).not_to eq(ancestor_b.output_size)
    #       expect(ancestor_a.input_size).to eq(ancestor_b.input_size)

    #       expect { my_breed_manager.breed(ancestor_a, ancestor_b) }.to raise_error("Parents must be the same output_size and input_size")
    #     end
    #   end
    #   context "when parents have differing input_size" do
    #     it "raises" do
    #       ancestor_a = ancestor_1
    #       ancestor_b = ancestor_4

    #       expect(ancestor_a.output_size).to eq(ancestor_b.output_size)
    #       expect(ancestor_a.input_size).not_to eq(ancestor_b.input_size)

    #       expect { my_breed_manager.breed(ancestor_a, ancestor_b) }.to raise_error("Parents must be the same output_size and input_size")
    #     end
    #   end
    #   context "when parents have differing output_size and input_size" do
    #     it "raises" do
    #       ancestor_a = ancestor_1
    #       ancestor_b = ancestor_5

    #       expect(ancestor_a.output_size).not_to eq(ancestor_b.output_size)
    #       expect(ancestor_a.input_size).not_to eq(ancestor_b.input_size)

    #       expect { my_breed_manager.breed(ancestor_a, ancestor_b) }.to raise_error("Parents must be the same output_size and input_size")
    #     end
    #   end

    #   context "children have expected values for" do
    #     let(child_1) {
    #       # cain
    #       child = my_breed_manager.breed(ancestor_adam, ancestor_eve, delta: delta_child_1)
    #       child.name = "Cain, child of #{ancestor_adam.name} and #{ancestor_eve.name}"
    #       child
    #     }

    #     let(ancestor_adam_json) { JSON.parse(ancestor_adam.to_json) }
    #     let(ancestor_eve_json) { JSON.parse(ancestor_eve.to_json) }
    #     let(child_1_json) { JSON.parse(child_1.to_json) }

    #     describe "#to_json" do
    #       context "for a new MiniNetManager with two initial 'ancestors' and one 'child'" do
    #         context "correctly exports" do
    #           it "the whole initial object" do
    #             expect(ancestor_adam).not_to be_nil
    #             expect(ancestor_eve).not_to be_nil
    #             expect(child_1).not_to be_nil

    #             # my_breed_manager.counter.inc("foo")
    #             counter = my_breed_manager.counter
    #             puts_debug
    #             puts_debug "counter.to_json: #{counter.to_json}"
    #             puts_debug
    #             puts_debug "my_breed_manager.to_json: #{my_breed_manager.to_json}"
    #             puts_debug

    #             # NOTE: 'exported' vs 'expected'
    #             exported_json = my_breed_manager.to_json
    #             expected_json = "{}"

    #             expect(exported_json).to be_a(String)
    #             expect(exported_json).to eq(expected_json)
    #           end
    #         end
    #       end
    #     end

    #     describe "#parts_to_copy stay the same, such as" do
    #       it "misc instance variables" do
    #         puts_debug
    #         puts_debug "  ancestor_adam.to_json: #{ancestor_adam.to_json}"
    #         puts_debug "  ancestor_eve.to_json: #{ancestor_eve.to_json}"
    #         puts_debug "  child_1.to_json: #{child_1.to_json}"
    #         puts_debug

    #         [
    #           "output_size", "input_size", "height_considering_bias", "width_indexes", "height_indexes",
    #           "learning_style", "disable_bias", "outputs_expected",
    #         ].each do |var|
    #           puts_debug
    #           puts_debug "var: #{var}"
    #           puts_debug
    #           expect(ancestor_adam_json[var]).to eq(ancestor_eve_json[var])
    #           expect(child_1_json[var]).to eq(ancestor_adam_json[var])
    #         end
    #       end
    #     end

    #     describe "#mix_parts are delta-mixed, such as" do
    #       it "bias_default" do
    #         ancestor_adam_value = ancestor_adam.bias_default
    #         ancestor_eve_value = ancestor_eve.bias_default
    #         expected_child_1_value = my_breed_manager.mix_nested_parts(ancestor_adam_value, ancestor_eve_value, delta_child_1)

    #         expect(ancestor_adam.bias_default).not_to eq(ancestor_eve.bias_default)
    #         expect(child_1.bias_default).to eq(expected_child_1_value)
    #       end

    #       it "learning_rate" do
    #         ancestor_adam_value = ancestor_adam.learning_rate
    #         ancestor_eve_value = ancestor_eve.learning_rate
    #         expected_child_1_value = my_breed_manager.mix_nested_parts(ancestor_adam_value, ancestor_eve_value, delta_child_1)

    #         expect(ancestor_adam.learning_rate).not_to eq(ancestor_eve.learning_rate)
    #         expect(child_1.learning_rate).to eq(expected_child_1_value)
    #       end

    #       it "momentum" do
    #         ancestor_adam_value = ancestor_adam.momentum
    #         ancestor_eve_value = ancestor_eve.momentum
    #         expected_child_1_value = my_breed_manager.mix_nested_parts(ancestor_adam_value, ancestor_eve_value, delta_child_1)

    #         expect(ancestor_adam.momentum).not_to eq(ancestor_eve.momentum)
    #         expect(child_1.momentum).to eq(expected_child_1_value)
    #       end

    #       it "deriv_scale" do
    #         ancestor_adam_value = ancestor_adam.deriv_scale
    #         ancestor_eve_value = ancestor_eve.deriv_scale
    #         expected_child_1_value = my_breed_manager.mix_nested_parts(ancestor_adam_value, ancestor_eve_value, delta_child_1)

    #         expect(ancestor_adam.deriv_scale).not_to eq(ancestor_eve.deriv_scale)
    #         expect(child_1.deriv_scale).to eq(expected_child_1_value)
    #       end

    #       it "weights" do
    #         ancestor_adam_value = ancestor_adam.weights
    #         ancestor_eve_value = ancestor_eve.weights
    #         expected_child_1_value = my_breed_manager.mix_nested_parts(ancestor_adam_value, ancestor_eve_value, delta_child_1)

    #         expect(ancestor_adam.weights).not_to eq(ancestor_eve.weights)
    #         expect(child_1.weights).to eq(expected_child_1_value)
    #       end

    #       it "inputs_given" do
    #         ancestor_adam_value = ancestor_adam.inputs_given
    #         ancestor_eve_value = ancestor_eve.inputs_given
    #         expected_child_1_value = my_breed_manager.mix_nested_parts(ancestor_adam_value, ancestor_eve_value, delta_child_1)

    #         expect(ancestor_adam.inputs_given).not_to eq(ancestor_eve.inputs_given)
    #         expect(child_1.inputs_given).to eq(expected_child_1_value)
    #       end

    #       it "outputs_guessed" do
    #         ancestor_adam_value = ancestor_adam.outputs_guessed
    #         ancestor_eve_value = ancestor_eve.outputs_guessed

    #         expected_child_1_value = my_breed_manager.mix_nested_parts(ancestor_adam_value, ancestor_eve_value, delta_child_1)

    #         expect(ancestor_adam.outputs_guessed).not_to eq(ancestor_eve.outputs_guessed)
    #         expect(child_1.outputs_guessed).to eq(expected_child_1_value)
    #       end

    #       it "output_deltas" do
    #         ancestor_adam_value = ancestor_adam.output_deltas
    #         ancestor_eve_value = ancestor_eve.output_deltas
    #         expected_child_1_value = my_breed_manager.mix_nested_parts(ancestor_adam_value, ancestor_eve_value, delta_child_1)

    #         expect(ancestor_adam.output_deltas).not_to eq(ancestor_eve.output_deltas)
    #         expect(child_1.output_deltas).to eq(expected_child_1_value)
    #       end

    #       it "last_changes" do
    #         ancestor_adam_value = ancestor_adam.last_changes
    #         ancestor_eve_value = ancestor_eve.last_changes
    #         expected_child_1_value = my_breed_manager.mix_nested_parts(ancestor_adam_value, ancestor_eve_value, delta_child_1)

    #         expect(ancestor_adam.last_changes).not_to eq(ancestor_eve.last_changes)
    #         expect(child_1.last_changes).to eq(expected_child_1_value)
    #       end

    #       it "output_errors" do
    #         ancestor_adam_value = ancestor_adam.output_errors
    #         ancestor_eve_value = ancestor_eve.output_errors
    #         expected_child_1_value = my_breed_manager.mix_nested_parts(ancestor_adam_value, ancestor_eve_value, delta_child_1)

    #         expect(ancestor_adam.output_errors).not_to eq(ancestor_eve.output_errors)
    #         expect(child_1.output_errors).to eq(expected_child_1_value)
    #       end

    #       it "input_deltas" do
    #         ancestor_adam_value = ancestor_adam.input_deltas
    #         ancestor_eve_value = ancestor_eve.input_deltas
    #         expected_child_1_value = my_breed_manager.mix_nested_parts(ancestor_adam_value, ancestor_eve_value, delta_child_1)

    #         expect(ancestor_adam.input_deltas).not_to eq(ancestor_eve.input_deltas)
    #         expect(child_1.input_deltas).to eq(expected_child_1_value)
    #       end
    #     end

    #     context "NOT copied and NOT mixed, but instead freshly initialized" do
    #       it "foo debug" do
    #         puts_debug
    #         puts_debug "ancestor_adam: #{ancestor_adam.to_json}"
    #         puts_debug
    #         puts_debug "ancestor_eve: #{ancestor_eve.to_json}"
    #         puts_debug
    #         puts_debug "child_1: #{child_1.to_json}"
    #         puts_debug
    #       end

    #       context "error_stats" do
    #         it "history_size" do
    #           # TODO: Adjust to compare error_stats
    #           ancestor_adam_value = ancestor_adam.error_stats.history_size
    #           ancestor_eve_value = ancestor_eve.error_stats.history_size
    #           # expected_child_1_value = my_breed_manager.mix_nested_parts(ancestor_adam_value, ancestor_eve_value, delta_child_1)
    #           expected_child_1_value = ancestor_adam_value

    #           expect(ancestor_adam_value).to eq(ancestor_eve_value)
    #           expect(child_1.error_stats.history_size).to eq(expected_child_1_value)
    #         end

    #         it "distance" do
    #           # TODO: Adjust to compare error_stats
    #           ancestor_adam_value = ancestor_adam.error_stats.distance
    #           ancestor_eve_value = ancestor_eve.error_stats.distance
    #           # expected_child_1_value = my_breed_manager.mix_nested_parts(ancestor_adam_value, ancestor_eve_value, delta_child_1)
    #           expected_child_1_value = -1.0

    #           expect(ancestor_adam_value).not_to eq(ancestor_eve_value)
    #           expect(child_1.error_stats.distance).to eq(expected_child_1_value)
    #         end

    #         it "history.size" do
    #           # TODO: Adjust to compare error_stats
    #           ancestor_adam_value = ancestor_adam.error_stats.history.size
    #           ancestor_eve_value = ancestor_eve.error_stats.history.size
    #           # expected_child_1_value = my_breed_manager.mix_nested_parts(ancestor_adam_value, ancestor_eve_value, delta_child_1)
    #           expected_child_1_value = 0

    #           expect(ancestor_adam_value).to eq(ancestor_eve_value)
    #           expect(child_1.error_stats.history.size).to eq(expected_child_1_value)
    #         end

    #         it "score" do
    #           # TODO: Adjust to compare error_stats
    #           ancestor_adam_value = ancestor_adam.error_stats.score
    #           ancestor_eve_value = ancestor_eve.error_stats.score
    #           # expected_child_1_value = my_breed_manager.mix_nested_parts(ancestor_adam_value, ancestor_eve_value, delta_child_1)
    #           expected_child_1_value = 1.8446744073709552e+19

    #           expect(ancestor_adam_value).not_to eq(ancestor_eve_value)
    #           expect(child_1.error_stats.score).to eq(expected_child_1_value)
    #         end
    #       end
    #     end
    #   end
    # end

    it "learning rates of Adam and Eve are different" do
      expect(ancestor_adam.learning_rate).to eq(ancestor_adam_learning_rate_expected)
      expect(ancestor_eve.learning_rate).to eq(ancestor_eve_learning_rate_expected)
      expect(ancestor_adam.learning_rate).not_to eq(ancestor_eve.learning_rate)
    end

    context "birth_id's are in the consistent order (when birthed in order" do
      it "first Adam" do
        expected_birth_counter = 0

        # Adam
        expected_birth_counter += 1
        expect(ancestor_adam.birth_id).to eq(expected_birth_counter)

        puts_debug
        puts_debug "ancestor_adam: #{ancestor_adam.to_json}"
      end

      it "first Adam then Eve" do
        expected_birth_counter = 0

        # Adam
        expected_birth_counter += 1
        expect(ancestor_adam.birth_id).to eq(expected_birth_counter)

        # Eve
        expected_birth_counter += 1
        expect(ancestor_eve.birth_id).to eq(expected_birth_counter)

        puts_debug
        puts_debug "ancestor_adam: #{ancestor_adam.to_json}"
        puts_debug
        puts_debug "ancestor_eve: #{ancestor_eve.to_json}"
      end

      it "first Adam then Eve followed by Cain" do
        expected_birth_counter = 0

        # Adam
        expected_birth_counter += 1
        expect(ancestor_adam.birth_id).to eq(expected_birth_counter)

        # Eve
        expected_birth_counter += 1
        expect(ancestor_eve.birth_id).to eq(expected_birth_counter)

        # Cain
        expected_birth_counter += 1
        child_1 = my_breed_manager.breed(ancestor_adam, ancestor_eve, delta: delta_child_1)
        child_1.name = "Cain, child of #{child_1.name} and #{ancestor_eve.name}"
        expect(child_1.birth_id).to eq(expected_birth_counter)

        puts_debug
        puts_debug "ancestor_adam: #{ancestor_adam.to_json}"
        puts_debug
        puts_debug "ancestor_eve: #{ancestor_eve.to_json}"
        puts_debug
        puts_debug "child_1: #{child_1.to_json}"
      end

      it "first Adam then Eve followed by Cain and then Abel" do
        expected_birth_counter = 0

        # Adam
        expected_birth_counter += 1
        expect(ancestor_adam.birth_id).to eq(expected_birth_counter)

        # Eve
        expected_birth_counter += 1
        expect(ancestor_eve.birth_id).to eq(expected_birth_counter)

        # Cain
        expected_birth_counter += 1
        child_1 = my_breed_manager.breed(ancestor_adam, ancestor_eve, delta: delta_child_1)
        child_1.name = "Cain, child of #{child_1.name} and #{ancestor_eve.name}"
        expect(child_1.birth_id).to eq(expected_birth_counter)

        # Abel
        expected_birth_counter += 1
        child_2 = my_breed_manager.breed(ancestor_adam, ancestor_eve, delta: delta_child_2)
        child_2.name = "Abel, child of #{child_2.name} and #{ancestor_eve.name}"
        expect(child_2.birth_id).to eq(expected_birth_counter)

        puts_debug
        puts_debug "ancestor_adam: #{ancestor_adam.to_json}"
        puts_debug
        puts_debug "ancestor_eve: #{ancestor_eve.to_json}"
        puts_debug
        puts_debug "child_1: #{child_1.to_json}"
        puts_debug
        puts_debug "child_2: #{child_2.to_json}"
      end
    end

    describe "#mix_parts" do
      context "first Adam then Eve followed by Cain" do
        let(child_1) { my_breed_manager.breed(ancestor_adam, ancestor_eve, delta: delta_child_1) }

        let(learning_rate_expected_1) {
          parent_a_part = ancestor_adam.learning_rate
          parent_b_part = ancestor_eve.learning_rate

          vector_a_to_b = parent_b_part - parent_a_part
          parent_a_part + (delta_child_1 * vector_a_to_b)
        }

        let(momentum_expected_1) {
          parent_a_part = ancestor_adam.momentum
          parent_b_part = ancestor_eve.momentum

          vector_a_to_b = parent_b_part - parent_a_part
          parent_a_part + (delta_child_1 * vector_a_to_b)
        }

        before_each do
          # Force variable to be initialized in specific order

          # Adam
          ancestor_adam.name = ancestor_adam.name + ""

          # Eve
          ancestor_eve.name = ancestor_eve.name + ""

          # Cain
          child_1.name = "Cain, child of #{child_1.name} and #{ancestor_eve.name}"
        end

        it "debug" do
          puts_debug
          puts_debug "ancestor_adam: #{ancestor_adam.to_json}"
          puts_debug
          puts_debug "ancestor_eve: #{ancestor_eve.to_json}"
          puts_debug
          puts_debug "child_1: #{child_1.to_json}"
          puts_debug

          # puts_debug
          # puts_debug "ancestor_adam: #{ancestor_adam.to_pretty_json}"
          # puts_debug
          # puts_debug "ancestor_eve: #{ancestor_eve.to_pretty_json}"
          # puts_debug
          # puts_debug "child_1: #{child_1.to_pretty_json}"
          # puts_debug
        end

        context "first child" do
          context "gets expected value(s) for" do
            it "learning_rate" do
              expect(child_1.learning_rate).to eq(learning_rate_expected_1)
            end

            it "momentum" do
              expect(child_1.momentum).to eq(momentum_expected_1)
            end

            # ... do likewise for other applicable variables

            context "mini_net_set" do
              let(li) { 0 }
              let(ti) { li }

              let(ancestor_adam_mini_net) { ancestor_adam.mini_net_set[li][ti] }
              let(ancestor_eve_mini_net) { ancestor_eve.mini_net_set[li][ti] }
              let(child_1_mini_net) { child_1.mini_net_set[li][ti] }

              context "gets expected value(s) for" do
                it "learning_rate" do
                  expect(ancestor_adam.learning_rate).to eq(ancestor_adam_mini_net.learning_rate)
                  expect(ancestor_eve.learning_rate).to eq(ancestor_eve_mini_net.learning_rate)
                  expect(child_1.learning_rate).to eq(child_1_mini_net.learning_rate)

                  expect(child_1_mini_net.learning_rate).to eq(learning_rate_expected_1)
                end

                it "momentum" do
                  # expect(child_1.momentum).to eq(momentum_expected_1)

                  expect(ancestor_adam.momentum).to eq(ancestor_adam_mini_net.momentum)
                  expect(ancestor_eve.momentum).to eq(ancestor_eve_mini_net.momentum)
                  expect(child_1.momentum).to eq(child_1_mini_net.momentum)

                  expect(child_1_mini_net.momentum).to eq(momentum_expected_1)
                end

                # ... do likewise for other applicable variables
              end
            end
          end

          context "does not get exact copy of either parent for values of" do
            it "learning_rate" do
              expect(child_1.learning_rate).not_to eq(ancestor_adam)
              expect(child_1.learning_rate).not_to eq(ancestor_eve)
            end

            it "momentum" do
              expect(child_1.momentum).not_to eq(ancestor_adam)
              expect(child_1.momentum).not_to eq(ancestor_eve)
            end

            # ... do likewise for other applicable variables

            context "mini_net_set" do
              let(li) { 0 }
              let(ti) { li }

              let(ancestor_adam_mini_net) { ancestor_adam.mini_net_set[li][ti] }
              let(ancestor_eve_mini_net) { ancestor_eve.mini_net_set[li][ti] }
              let(child_1_mini_net) { child_1.mini_net_set[li][ti] }

              context "does not get exact copy of either parent for values of" do
                it "learning_rate" do
                  expect(child_1_mini_net.learning_rate).not_to eq(ancestor_adam_mini_net)
                  expect(child_1_mini_net.learning_rate).not_to eq(ancestor_eve_mini_net)
                end

                it "momentum" do
                  expect(child_1_mini_net.momentum).not_to eq(ancestor_adam_mini_net)
                  expect(child_1_mini_net.momentum).not_to eq(ancestor_eve_mini_net)
                end

                # ... do likewise for other applicable variables
              end
            end
          end
        end
      end
    end
  end
end
