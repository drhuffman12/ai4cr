require "./file_type"

module Ai4cr
  module Utils
    module IoData
      abstract class Abstract
        property file_path : String
        property raw = ""
        property iod = Array(Array(Float64)).new
        property prefix_raw_qty
        property prefix_raw_char

        def initialize(@file_path : String, file_content_type : FileType, @prefix_raw_qty = 0, @prefix_raw_char = " ")
          case file_content_type
          when FileType::Raw
            @raw = (prefix_raw_char * prefix_raw_qty) + File.read(file_path)
            @iod = self.class.convert_raw_to_iod(@raw)
          when FileType::Iod
            contents = File.read(file_path)
            @iod = Array(Array(Float64)).from_json(contents)
            @raw = self.class.convert_iod_to_raw(@iod)[prefix_raw_qty..-1]
          end
        end

        def self.convert_raw_to_iod(raw) : Array(Array(Float64))
          raise "Must be implemented in subclass"
        end

        def self.convert_iod_to_raw(iod) : String
          raise "Must be implemented in subclass"
        end

        def save_raw(to_file_path : String)
          File.write(to_file_path, raw)
        end

        def save_iod(to_file_path : String)
          File.write(to_file_path, iod.to_json)
        end

        def iod_uncertainty(iod_guessed)
          return 1.0 if iod_guessed.nil? || iod_guessed.empty?
          # NOTE: This is NOT the same as accuracy! It is just a score of how certain the net is about the guess.
          deltas = iod_guessed.flatten.map do |guess|
            if guess.round >= 0.5
              1.0 - guess
            else
              -guess
            end
          end
          deltas.map { |d| d.abs }.sum / deltas.size
        end

        def iod_certainty(iod_guessed)
          1.0 - iod_uncertainty(iod_guessed)
        end

        ####
        # Below is for AI algorythms requiring multiple time-column inputs/outputs

        def iod_to_io_set_with_offset(offset : Int32)
          io_set_qty = iod.size - offset

          input_i_max = io_set_qty - 1
          output_i_max = iod.size - 1

          inputs = iod[0..input_i_max].map { |d| d }
          outputs = iod[offset..output_i_max].map { |d| d }

          {
            inputs:  inputs,
            outputs: outputs,
          }
        end

        def iod_to_io_set_with_offset_time_cols(time_cols : Int32, offset : Int32)
          io_set = iod_to_io_set_with_offset(offset)

          input_i_max = io_set[:inputs].size - 1
          output_i_max = io_set[:outputs].size - 1

          raise "io size mismatch" if input_i_max != output_i_max

          tc_set_qty = input_i_max - time_cols + 1
          tc_set_indexes = Array.new(tc_set_qty) { |i| i }

          input_set = tc_set_indexes.map do |si|
            io_set[:inputs][si..si + time_cols - 1]
          end

          output_set = tc_set_indexes.map do |si|
            io_set[:outputs][si..si + time_cols - 1]
          end

          {
            input_set:  input_set,
            output_set: output_set,
          }
        end
      end
    end
  end
end
