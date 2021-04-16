require "./../../../../../spectator_helper"

Spectator.describe Ai4cr::NeuralNetwork::Rnn::RnnSimpleConcerns::CalcGuess do
  let(rnn_simple) { Ai4cr::NeuralNetwork::Rnn::RnnSimple.new }

  let(input_set_given) {
    [
      [0.1, 0.2],
      [0.3, 0.4],
    ]
  }

  describe "#eval" do
    let(expected_outputs_guessed_before) { [[0.0], [0.0]] }

    context "with hard-coded weights" do
      let(expected_outputs_guessed) {
        [[0.119], [0.09780000000000003]]
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

      context "before" do
        it "outputs_guessed start off all zero's" do
          expect(rnn_simple.outputs_guessed).to eq(expected_outputs_guessed_before)
        end
      end

      context "during" do
        pending "calls 'step_load_inputs'" do
          expect(rnn_simple).to receive(:step_load_inputs)

          rnn_simple.eval(input_set_given)
        end

        pending "calls 'step_calc_forward'" do
          expect(rnn_simple).to receive(:step_calc_forward)

          rnn_simple.eval(input_set_given)
        end
      end

      context "after" do
        it "calculates expected outputs" do
          rnn_simple.eval(input_set_given)

          assert_approximate_equality_of_nested_list(expected_outputs_guessed, rnn_simple.outputs_guessed)
        end
      end
    end
  end

  describe "#all_mini_net_outputs" do
    context "with hard-coded weights" do
      let(rnn_simple) { Ai4cr::NeuralNetwork::Rnn::RnnSimple.new }

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

          assert_approximate_equality_of_nested_list(expected_all_mini_net_outputs_after, rnn_simple.all_mini_net_outputs)
        end
      end
    end
  end
end
