require "./../../../../../spec_helper"
require "./../../../../../spectator_helper"

Spectator.describe Ai4cr::NeuralNetwork::Cmn::RnnConcerns::TrainAndAdjust do
  let(deriv_scale) { 0.1 }
  let(learning_rate) { 0.2 }
  let(momentum) { 0.3 }
  let(rnn_simple) {
    Ai4cr::NeuralNetwork::Cmn::RnnSimple.new(
      deriv_scale: deriv_scale,
      learning_rate: learning_rate,
      momentum: momentum,
    )
  }

  let(input_set_given) {
    [
      [0.1, 0.2],
      [0.3, 0.4],
    ]
  }
  let(delta_1_thousandths) { 0.000_1 }

  describe "#train" do
    let(expected_outputs_guessed_before) { [[0.0], [0.0]] }
    let(expected_outputs_trained) { [[0.4], [0.6]] }

    context "with hard-coded weights" do
      let(expected_outputs_guessed) {
        [[0.14193], [0.0]]
      }
      let(hard_coded_weights) {
        [
          [
            [
              [-0.4, -0.3, -0.2],
              [-0.1, 0.05, 0.1],
              [0.2, 0.3, 0.4],
            ],
            [
              [-0.41, -0.31, -0.21],
              [-0.11, 0.06, 0.11],
              [0.21, 0.31, 0.41],
              [-0.42, -0.32, -0.22],
              [-0.12, 0.07, 0.12],
              [0.22, 0.32, 0.42],
            ],
          ],
          [
            [
              [-0.23],
              [0.13],
              [0.33],
            ],
            [
              [-0.44],
              [-0.24],
              [0.24],
              [0.44],
            ],
          ],
        ]
      }

      before_each do
        rnn_simple.synaptic_layer_indexes.map do |li|
          rnn_simple.time_col_indexes.map do |ti|
            # NOTE: We 'clone' the values so that our tests don't affect the original 'hard_coded_weights'!
            rnn_simple.mini_net_set[li][ti].weights = hard_coded_weights[li][ti].clone
          end
        end
      end

      context "before" do
        it "outputs_guessed start off all zero's" do
          expect(rnn_simple.outputs_guessed).to eq(expected_outputs_guessed_before)
        end
      end

      context "during" do
        pending "calls 'eval(input_set_given)'" do
          # TODO: un-pend after '...to receive...' is fixed
          expect(rnn_simple).to receive(:eval).with(input_set_given)

          rnn_simple.train(input_set_given, expected_outputs_trained)
        end
      end

      context "after" do
        let(expected_error_total) { 0.01680627208738801 }
        let(expected_all_output_errors) {
          [
            [
              [-0.05086850000000001, 0.0797935, 0.0629835],
              [-0.264, -0.144, 0.144] # , 0.264]
            ],
            [
              [0.2639],
              [0.6]
            ]
          ]
        }

        it "guesses expected outputs" do
          rnn_simple.train(input_set_given, expected_outputs_trained)

          assert_approximate_equality_of_nested_list(expected_outputs_guessed, rnn_simple.outputs_guessed)
        end

        it "sets expected all_output_errors" do
          rnn_simple.train(input_set_given, expected_outputs_trained)
          all_output_errors = rnn_simple.all_output_errors

          expect(all_output_errors).to eq(expected_all_output_errors)
        end

        it "calculates output_deltas" do
          before_all_mini_net_output_deltas = rnn_simple.all_mini_net_output_deltas.clone

          rnn_simple.train(input_set_given, expected_outputs_trained)

          after_all_mini_net_output_deltas = rnn_simple.all_mini_net_output_deltas.clone

          assert_approximate_inequality_of_nested_list(before_all_mini_net_output_deltas, after_all_mini_net_output_deltas, delta_1_thousandths)
        end

        it "calculates input_deltas" do
          before_all_mini_net_input_deltas = rnn_simple.all_mini_net_input_deltas.clone

          rnn_simple.train(input_set_given, expected_outputs_trained)

          after_all_mini_net_input_deltas = rnn_simple.all_mini_net_input_deltas.clone

          assert_approximate_inequality_of_nested_list(after_all_mini_net_input_deltas, before_all_mini_net_input_deltas, delta_1_thousandths)
        end

        it "adjusts weights" do
          rnn_simple.train(input_set_given, expected_outputs_trained)

          assert_approximate_inequality_of_nested_list(hard_coded_weights, rnn_simple.all_mini_net_weights, delta_1_thousandths)
        end

        it "caches last_changes" do
          before_all_mini_net_last_changes = rnn_simple.all_mini_net_last_changes.clone

          rnn_simple.train(input_set_given, expected_outputs_trained)

          after_all_mini_net_last_changes = rnn_simple.all_mini_net_last_changes.clone

          assert_approximate_inequality_of_nested_list(before_all_mini_net_last_changes, after_all_mini_net_last_changes, delta_1_thousandths)
        end

        it "calcs expected all_output_errors" do
          rnn_simple.train(input_set_given, expected_outputs_trained)
          all_output_errors = rnn_simple.all_output_errors

          expect(all_output_errors).to eq(expected_all_output_errors)
        end

        it "returns expected error_total" do
          error_total = rnn_simple.train(input_set_given, expected_outputs_trained)

          expect(error_total).to eq(expected_error_total)
        end
      end

      context "after 2nd training session" do
        # let(rnn_simple) { Ai4cr::NeuralNetwork::Cmn::RnnSimple.new }

        let(expected_outputs_guessed_2nd) {
          [[0.1748345741196996], [0.10429156621331176]]
        }
        let(expected_error_total) { 0.01680627208738801 }
        let(expected_error_total_2nd) { 0.00800202291515254 }
        let(expected_all_output_errors) {
          [
            [
              [-0.05086850000000001, 0.0797935, 0.0629835],
              [-0.264, -0.144, 0.144]
            ],
            [
              [0.2639],
              [0.6]
            ]
          ]
        }
        let(expected_all_output_errors_2nd) {
          # [
          #   [
          #     [0.0025360797083354286, 0.010592034318506409, 0.02973469735998057],
          #     [-0.025985534533164822, 0.005527280460422763, 0.055271111289300526]
          #   ],
          #   [
          #     [0.092503233903734855],
          #     [0.09353174500913009]
          #   ]
          # ]

          [
            [
              [-0.015259420733033626, 0.056563972082894265, 0.07359704509032028],
              [-0.20026620724982205, -0.06840776386256296, 0.18111996470324498]
            ],
            [
              [0.24553369771547515],
              [0.4957084337866882]
            ]
          ]
        }

        it "guesses expected outputs" do
          rnn_simple.train(input_set_given, expected_outputs_trained)
          rnn_simple.train(input_set_given, expected_outputs_trained)

          # assert_approximate_equality_of_nested_list(expected_outputs_guessed_2nd, rnn_simple.outputs_guessed)
          expect(rnn_simple.outputs_guessed).to eq(expected_outputs_guessed_2nd)
        end

        it "calculates output_deltas" do
          before_all_mini_net_output_deltas = rnn_simple.all_mini_net_output_deltas.clone

          rnn_simple.train(input_set_given, expected_outputs_trained)

          mid_all_mini_net_output_deltas = rnn_simple.all_mini_net_output_deltas.clone

          rnn_simple.train(input_set_given, expected_outputs_trained)

          after_all_mini_net_output_deltas = rnn_simple.all_mini_net_output_deltas.clone

          assert_approximate_inequality_of_nested_list(before_all_mini_net_output_deltas, mid_all_mini_net_output_deltas, delta_1_thousandths)
          assert_approximate_inequality_of_nested_list(mid_all_mini_net_output_deltas, after_all_mini_net_output_deltas, delta_1_thousandths)
        end

        it "calculates input_deltas" do
          before_all_mini_net_input_deltas = rnn_simple.all_mini_net_input_deltas.clone

          rnn_simple.train(input_set_given, expected_outputs_trained)

          mid_all_mini_net_input_deltas = rnn_simple.all_mini_net_input_deltas.clone

          rnn_simple.train(input_set_given, expected_outputs_trained)

          after_all_mini_net_input_deltas = rnn_simple.all_mini_net_input_deltas.clone

          assert_approximate_inequality_of_nested_list(before_all_mini_net_input_deltas, mid_all_mini_net_input_deltas, delta_1_thousandths)
          assert_approximate_inequality_of_nested_list(mid_all_mini_net_input_deltas, after_all_mini_net_input_deltas, delta_1_thousandths)
        end

        it "adjusts weights" do
          rnn_simple.train(input_set_given, expected_outputs_trained)
          mid_all_mini_net_input_deltas = rnn_simple.all_mini_net_weights.clone

          rnn_simple.train(input_set_given, expected_outputs_trained)
          after_all_mini_net_input_deltas = rnn_simple.all_mini_net_weights.clone

          assert_approximate_inequality_of_nested_list(hard_coded_weights, mid_all_mini_net_input_deltas, delta_1_thousandths)
          assert_approximate_inequality_of_nested_list(mid_all_mini_net_input_deltas, after_all_mini_net_input_deltas, delta_1_thousandths)
        end

        it "caches last_changes" do
          before_all_mini_net_last_changes = rnn_simple.all_mini_net_last_changes.clone

          rnn_simple.train(input_set_given, expected_outputs_trained)

          mid_all_mini_net_last_changes = rnn_simple.all_mini_net_last_changes.clone

          rnn_simple.train(input_set_given, expected_outputs_trained)

          after_all_mini_net_last_changes = rnn_simple.all_mini_net_last_changes.clone

          assert_approximate_inequality_of_nested_list(before_all_mini_net_last_changes, mid_all_mini_net_last_changes, delta_1_thousandths)
          assert_approximate_inequality_of_nested_list(mid_all_mini_net_last_changes, after_all_mini_net_last_changes, delta_1_thousandths)
        end

        it "calcs expected all_output_errors" do
          rnn_simple.train(input_set_given, expected_outputs_trained)
          mid_output_errors = rnn_simple.all_output_errors
          expect(mid_output_errors).to eq(expected_all_output_errors) # TODO

          rnn_simple.train(input_set_given, expected_outputs_trained)
          after_output_errors = rnn_simple.all_output_errors
          expect(after_output_errors).to eq(expected_all_output_errors_2nd)
        end

        it "returns expected error_total" do
          mid_error_total = rnn_simple.train(input_set_given, expected_outputs_trained)
          expect(mid_error_total).to eq(expected_error_total)

          after_error_total = rnn_simple.train(input_set_given, expected_outputs_trained)
          expect(after_error_total).to eq(expected_error_total_2nd)
        end
      end
    end
  end

  describe "#all_mini_net_outputs" do
    context "with hard-coded weights" do
      let(rnn_simple) { Ai4cr::NeuralNetwork::Cmn::RnnSimple.new }

      let(input_set_given) {
        [
          [0.1, 0.2],
          [0.3, 0.4],
        ]
      }

      let(expected_all_mini_net_outputs_before) {
        [
          [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
          ],
          [
            [0.0],
            [0.0],
          ],
        ]
      }

      let(expected_all_mini_net_outputs_after) {
        # TODO: manually verify calc's.
        #   For now, we'll assume accuracy and move onto
        #   verifying via some training sessions with
        #   some 'real data'
        [
          [
            [0.14, 0.27, 0.4],
            [0.0, 0.17099999999999999, 0.42200000000000004],
          ],
          [
            [0.119],
            [0.09780000000000003],
          ],
        ]
      }

      before_each do
        weights = [
          [
            [
              [-0.4, -0.3, -0.2],
              [-0.1, 0.0, 0.1],
              [0.2, 0.3, 0.4],
            ],
            [
              [-0.4, -0.3, -0.2],
              [-0.1, 0.0, 0.1],
              [0.2, 0.3, 0.4],
              [-0.4, -0.3, -0.2],
              [-0.1, 0.0, 0.1],
              [0.2, 0.3, 0.4],
            ],
          ],
          [
            [
              [-0.2],
              [0.1],
              [0.3],
            ],
            [
              [-0.4],
              [-0.2],
              [0.2],
              [0.4],
            ],
          ],
        ]
        rnn_simple.synaptic_layer_indexes.map do |li|
          rnn_simple.time_col_indexes.map do |ti|
            rnn_simple.mini_net_set[li][ti].weights = weights[li][ti]
          end
        end
      end

      context "before #eval" do
        it "returns all-zero outputs" do
          expect(rnn_simple.all_mini_net_outputs).to eq(expected_all_mini_net_outputs_before)
        end
      end

      context "after #eval" do
        it "returns expected non-zero outputs" do
          rnn_simple.eval(input_set_given)

          assert_approximate_equality_of_nested_list(expected_all_mini_net_outputs_after, rnn_simple.all_mini_net_outputs, delta_1_thousandths**3)
        end
      end
    end
  end
end
