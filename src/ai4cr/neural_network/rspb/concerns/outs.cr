# Ai4cr::NeuralNetwork::Rspb::Concerns::Outs
module Ai4cr
  module NeuralNetwork
    module Rspb
      # struct PrevMem
      module Concerns
        module Outs
          property outputs_size : Int32 = 0
          property outputs : Array(Float64) = [0.0]

          def init_outputs(outputs_new : IoDataSet? = nil)
            # validate_outputs_new(outputs_new)
            if outputs_new
              validate_outputs_new(outputs_new)
              # @outputs_size = outputs_new.size
              @outputs = outputs_new # .clone
            else
              @outputs = (1..outputs_size).map { 0.0 }
            end
            return self
          end

          def validate_outputs_new(outputs_new : IoDataSet)
            raise ArgumentError.new("Size Mismatch! Expected outputs_size: #{outputs_size}, given outputs_new.size: #{outputs_new.size}") if outputs_size != outputs_new.size
          end
        end
      end
    end
  end
end
