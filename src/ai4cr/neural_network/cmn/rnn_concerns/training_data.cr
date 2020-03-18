require "json"

module Ai4cr
  module NeuralNetwork
    module Cmn
      module RnnConcerns
        class TrainingData
          getter sequenced_inputs, output_offset, time_col_qty

          getter input_sets : Array(Array(Array(Float64)))
          getter output_sets : Array(Array(Array(Float64)))
          getter io_set_qty : Int32
          getter io_set_index_max : Int32
          getter io_set_index_ids : Array(Int32)

          def initialize(
            @sequenced_inputs : Array(Array(Float64)) = [[0.0, 0.1, 0.2], [1.0, 1.1, 1.2], [2.0, 2.1, 2.2]],
            @output_offset = 1,
            @time_col_qty = 2
          )
            # This assumes that you have:
            # * sis = max time cols (aka sequenced_inputs.size)
            # * oo = offset of inputs to outputs (aka output_offset)
            # * tcq = offset of inputs to outputs (aka time_col_qty)
            #
            # time col index | in sequence index | out seqence index
            # 0 | 0 | n/a
            # 1 | 1 | n/a
            # 2 | 2 | n/a
            # ... | ... | ...
            # oo | oo | 0
            # oo + 1 | oo + 1 | 1
            # oo + 2 | oo + 2 | 2
            # ... | ... | ...
            # oo + n | oo + n | n
            # oo + n + 1 | oo + n + 1 | n + 1
            # ... | ... | ...

            # sis - (oo + n + 1) | sis - (oo + n + 1) | sis - (n + 1)

            # sis - 2 | n/a | sis - (oo - tcq)
            # ... | n/a | sis - ...
            # sis - 2 | n/a | sis - (oo - 2)
            # sis - 1 | n/a | sis - (oo - 1)
            # sis | n/a | sis - (oo)

            @input_sets = Array(Array(Array(Float64))).new
            @output_sets = Array(Array(Array(Float64))).new
            @io_set_qty = sequenced_inputs.size - output_offset - (time_col_qty - 1)
            @io_set_index_max = @io_set_qty - 1
            @io_set_index_ids = (0..@io_set_index_max).to_a

            @io_set_index_ids.each do |i|
              input_index_from = i
              input_index_to = i + time_col_qty - 1 # output_offset

              output_index_from = output_offset + i
              output_index_to = output_offset + i + time_col_qty - 1 # output_offset

              input_set = sequenced_inputs[input_index_from..input_index_to]
              @input_sets << input_set

              output_set = sequenced_inputs[output_index_from..output_index_to]
              @output_sets << output_set
            end
          end
        end
      end
    end
  end
end
