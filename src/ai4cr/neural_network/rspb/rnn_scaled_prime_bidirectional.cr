require "./util"

module Ai4cr
  module NeuralNetwork
    module Rspb
      # Ai4cr::NeuralNetwork::RnnScaledPrimeBidirectional
      struct RnnScaledPrimeBidirectional
        # The below defaults probably should be moved to a config file
        PRIME_QTY_DEFAULT = ENV.has_key?("AI4CR_RSB_PRIME_QTY_DEFAULT") && ENV["AI4CR_RSB_PRIME_QTY_DEFAULT"].to_i32 > 0 ? ENV["AI4CR_RSB_PRIME_QTY_DEFAULT"].to_i32 : 8
        ZOOM_QTY_DEFAULT = ENV.has_key?("AI4CR_RSB_ZOOM_QTY_DEFAULT") && ENV["AI4CR_RSB_ZOOM_QTY_DEFAULT"].to_i32 > 0 ? ENV["AI4CR_RSB_ZOOM_QTY_DEFAULT"].to_i32 : 8
        PANEL_QTY_DEFAULT = ENV.has_key?("AI4CR_RSB_PANEL_QTY_DEFAULT") && ENV["AI4CR_RSB_PANEL_QTY_DEFAULT"].to_i32 > 0 ? ENV["AI4CR_RSB_PANEL_QTY_DEFAULT"].to_i32 : 2

        property prime_qty : Int32
        property zoom_qty : Int32
        property panel_qty : Int32

        property prime_offsets : Array(Int32)
        property zoom_scales : Array(Int32)
        property zoomed_prime_offsets : Array(Array(Int32))

        property io_min_qty_required : Int32

        def initialize(@prime_qty = PRIME_QTY_DEFAULT, @zoom_qty = ZOOM_QTY_DEFAULT, @panel_qty = PANEL_QTY_DEFAULT)
          @prime_offsets = Util.gen_prime_offsets(@prime_qty)
          @zoom_scales = Util.gen_zoom_scales(@zoom_qty)
          @zoomed_prime_offsets = Util.gen_zoomed_prime_offsets(@zoom_qty, @prime_qty)

          @io_min_qty_required = zoomed_prime_offsets.last.last * 2 + 1
        end
      end
    end
  end
end

# mkdir -p spec/ai4cr/neural_network/rnn_scaled_prime_bidirectional
# icr(0.24.2) > require "./src/ai4cr/neural_network/rnn_scaled_prime_bidirectional"
# icr(0.24.2) > net = Ai4cr::NeuralNetwork::RnnScaledPrimeBidirectional.new
#  => Ai4cr::NeuralNetwork::RnnScaledPrimeBidirectional(@prime_qty=8, @zoom_qty=8, @panel_qty=3, @prime_deltas=[1, 2, 3, 5, 8, 13, 21, 34], @prime_max=34, @zoom_deltas=[1, 2, 4, 8, 16, 32, 64, 128], @zoom_max=128, @time_col_size=13056)
