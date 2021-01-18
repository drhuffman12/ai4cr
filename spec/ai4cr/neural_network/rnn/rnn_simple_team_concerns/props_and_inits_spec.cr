require "./../../../../spec_helper"
require "./../../../../spectator_helper"

Spectator.describe Ai4cr::NeuralNetwork::Rnn::RnnSimpleTeam do
  context "Team of RNN's Contained Mini Nets" do
    describe "#initialize" do
      let(team_size) { 2 }

      let(rnn_simple_team) {
        Ai4cr::NeuralNetwork::Rnn::RnnSimpleTeam.new(
          team_size: team_size
        )
      }

      context "team_members is initialized such that" do
        context "values should match for" do
          it "io_offset" do
            expect(rnn_simple_team.team_members[0].io_offset).to eq(rnn_simple_team.team_members[1].io_offset)
          end

          it "time_col_qty" do
            expect(rnn_simple_team.team_members[0].time_col_qty).to eq(rnn_simple_team.team_members[1].time_col_qty)
          end

          it "input_size" do
            expect(rnn_simple_team.team_members[0].input_size).to eq(rnn_simple_team.team_members[1].input_size)
          end

          it "output_size" do
            expect(rnn_simple_team.team_members[0].output_size).to eq(rnn_simple_team.team_members[1].output_size)
          end

          it "hidden_size_given" do
            expect(rnn_simple_team.team_members[0].hidden_size_given).to eq(rnn_simple_team.team_members[1].hidden_size_given)
          end

          it "hidden_layer_qty" do
            expect(rnn_simple_team.team_members[0].hidden_layer_qty).to eq(rnn_simple_team.team_members[1].hidden_layer_qty)
          end

          it "hidden_size" do
            expect(rnn_simple_team.team_members[0].hidden_size).to eq(rnn_simple_team.team_members[1].hidden_size)
          end

          it "learning_style" do
            expect(rnn_simple_team.team_members[0].learning_style).to eq(rnn_simple_team.team_members[1].learning_style)
          end
        end

        context "values should just use same defaults (for now) for" do
          # NOTE: These are randomly generated values, so there should usually be some random difference, but it's not 100% guarantee.
          it "disable_bias" do
            expect(rnn_simple_team.team_members[0].disable_bias).to eq(rnn_simple_team.team_members[1].disable_bias)
          end

          it "bias_default" do
            expect(rnn_simple_team.team_members[0].bias_default).to eq(rnn_simple_team.team_members[1].bias_default)
          end
        end

        context "values should differ randomly for" do
          # NOTE: These are randomly generated values, so there should usually be some random difference, but it's not 100% guarantee.
          it "learning_rate" do
            expect(rnn_simple_team.team_members[0].learning_rate).not_to eq(rnn_simple_team.team_members[1].learning_rate)
          end

          it "momentum" do
            expect(rnn_simple_team.team_members[0].momentum).not_to eq(rnn_simple_team.team_members[1].momentum)
          end

          it "deriv_scale" do
            expect(rnn_simple_team.team_members[0].deriv_scale).not_to eq(rnn_simple_team.team_members[1].deriv_scale)
          end
        end

        context "hidden_size" do
          it "is an Int32" do
            expect(rnn_simple_team.hidden_size).to be_a(Int32)
          end

          it "is not nil" do
            expect(rnn_simple_team.hidden_size).not_to be_nil
          end
        end
      end
    end
  end
end
