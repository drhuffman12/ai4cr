require "./../spec_helper"
require "./../spectator_helper"

Spectator.describe Ai4cr::ErrorStats do
  let(given_history_size) { 8 }
  let(error_stats) { Ai4cr::ErrorStats.new(given_history_size) }

  let(expected_initial_distance) { -1.0 }

  let(expected_initial_score) {
    1.8446744073709552e+19 # Float64::MAX ** (1.0/16)
  }
  let(expected_initial_history) {
    [] of Float64
  }
  let(to_json) { error_stats.to_json }
  let(expected_initial_json) { "{\"history_size\":8,\"distance\":-1.0,\"history\":[],\"score\":1.8446744073709552e+19}" }
  let(expected_later_json) { "{\"history_size\":8,\"distance\":10.0,\"history\":[10.0],\"score\":5.0}" }

  describe "#initialize" do
    context "has" do
      it "given history_size" do
        expect(error_stats.history_size).to eq(given_history_size)
      end

      it "initial distance of zero" do
        expect(error_stats.distance).to eq(expected_initial_distance)
      end

      it "initial history of an empty Array(Float64)" do
        expect(error_stats.history).to eq(expected_initial_history)
      end

      it "initial history of an empty Array(Float64)" do
        expect(error_stats.score).to eq(expected_initial_score)
      end
    end

    context "to_json" do
      context "returns" do
        it "expected_initial_json" do
          expect(to_json).to eq(expected_initial_json)
        end
      end
    end

    context "from_json" do
      context "when given error_stats.to_json" do
        context "and re-exported to_json" do
          it "returns json matching original converted to_json" do
            expect(Ai4cr::ErrorStats.from_json(to_json).to_json).to eq(expected_initial_json)
          end
        end
      end
    end
  end

  describe "#distance(value)" do
    let(given_value) { 10.0 }
    it "sets 'distance' to given value" do
      error_stats.distance = given_value

      expect(error_stats.distance).to eq(given_value)
    end

    context "to_json" do
      context "returns" do
        it "expected_later_json" do
          error_stats.distance = given_value

          expect(to_json).to eq(expected_later_json)
        end
      end
    end

    context "from_json" do
      context "when given error_stats.to_json" do
        context "and re-exported to_json" do
          it "returns json matching original converted to_json" do
            error_stats.distance = given_value

            expect(Ai4cr::ErrorStats.from_json(to_json).to_json).to eq(expected_later_json)
          end
        end
      end
    end
  end

  describe "#plot_error_distance_history" do
    let(given_error_distance_history) { [4.0, 2.0, 3.0, 1.0, 0.5, 0.3, 0.1, 0.4, 0.01] }

    let(expected_plot_first_4) { "▴▴▴▴" }
    let(expected_plot_all) { "▴▴▴▄▂_▃▿" }

    context "when given_error_distance_history.size > history_size" do
      it "given_error_distance_history.size > history_size" do
        expect(given_error_distance_history.size).to be > error_stats.history_size
      end

      context "after loading less than history_size of given_error_distance_history" do
        before_each do
          given_error_distance_history[0..3].each do |value|
            error_stats.distance = value
          end
        end

        it "returns expected string" do
          plot = error_stats.plot_error_distance_history(in_bw: true).clone

          # puts plot

          expect(plot).to eq(expected_plot_first_4)
        end

        it "plot size is less than history_size" do
          plot = error_stats.plot_error_distance_history(in_bw: true).clone

          # puts plot
          expect(plot.size).to be < error_stats.history_size
        end
      end

      context "after loading at least history_size of given_error_distance_history" do
        before_each do
          given_error_distance_history.each do |value|
            error_stats.distance = value
          end
        end

        it "returns expected string" do
          plot = error_stats.plot_error_distance_history(in_bw: true).clone

          # puts plot

          expect(plot).to eq(expected_plot_all)
        end

        it "plot size maxes out at history_size" do
          plot = error_stats.plot_error_distance_history(in_bw: true).clone

          # puts plot
          expect(plot.size).to eq(error_stats.history_size)
        end
      end
    end
  end
end
