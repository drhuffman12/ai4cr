require "./file_type"

module Ai4cr
  module Utils
    module IoData
      abstract class Abstract
        # TODO: refactor (or add/rename methods) so we can go from one-char per time-column (in bits per tc's inputs)
        #   to multiple char's per time col (several per tc's inputs)
        #   REMINDER: Must consider Ai4cr::NeuralNetwork::Rnn::RnnSimpleConcerns::PropsAndInits::INPUT_SIZE_MIN and OUTPUT_SIZE_MIN

        property file_path : String
        property raw = ""
        property iod = Array(Array(Float64)).new
        property prefix_raw_qty
        property prefix_raw_char

        # TODO: Replace(?)/Supplement(?) below with actual UTF-to-ASCII conversion! (And ASCII-to-UTF reversion)
        property default_to_bit_size # e.g.: to allow forcing from 32bit utf down to 8bit ascii (ignoring higher bits)

        def initialize(
          @file_path : String,
          file_content_type : FileType,
          @prefix_raw_qty = 0,
          @prefix_raw_char = " ",
          @default_to_bit_size = 0
        )
          case file_content_type
          when FileType::Raw
            @raw = (prefix_raw_char * prefix_raw_qty) + File.read(file_path)
            @iod = self.class.convert_raw_to_iod(@raw, @default_to_bit_size)
          when FileType::Iod
            contents = File.read(file_path)
            @iod = Array(Array(Float64)).from_json(contents)
            @raw = self.class.convert_iod_to_raw(@iod, @default_to_bit_size)[prefix_raw_qty..-1]
          end
        end

        def self.convert_raw_to_iod(raw, default_to_bit_size = 0) : Array(Array(Float64))
          raise "Must be implemented in subclass"
        end

        def self.convert_iod_to_raw(iod, default_to_bit_size = 0) : String
          raise "Must be implemented in subclass"
        end

        def save_raw(to_file_path : String)
          File.write(to_file_path, raw)
        end

        def save_iod(to_file_path : String)
          File.write(to_file_path, iod.to_json)
        end

        def iod_uncertainty_per_guess(iod_guessed) : Array(Float64)
          return [1.0] if iod_guessed.nil? || iod_guessed.empty?
          # NOTE: This is NOT the same as accuracy! It is just a score of how certain the net is about the guess.
          iod_guessed.flatten.map do |guess|
            if guess.round >= 0.5
              1.0 - guess
            else
              -guess # guess - 0
            end.abs
          end
        end

        def iod_certainty_per_guess(iod_guessed) : Array(Float64)
          iod_uncertainty_per_guess(iod_guessed).map{|u| 1 - u}
        end

        def iod_uncertainty(iod_guessed)
          return 1.0 if iod_guessed.nil? || iod_guessed.empty?
          deltas = iod_uncertainty_per_guess(iod_guessed)
          deltas.sum(&.abs) / deltas.size
        end

        def iod_certainty(iod_guessed)
          # 1.0 - iod_uncertainty(iod_guessed)
          return 1.0 if iod_guessed.nil? || iod_guessed.empty?
          deltas = iod_certainty_per_guess(iod_guessed)
          deltas.sum(&.abs) / deltas.size
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

          # in_groups_of = Ai4cr::NeuralNetwork::Rnn::RnnSimpleConcerns::PropsAndInits::INPUT_SIZE_MIN

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

        def iod_to_io_set_with_offset_and_io_per_time_cols(time_cols : Int32, offset : Int32, io_size : Int32, io_offset : Int32 = -1)
          # input_set = [0.0]
          # output_set = [0.0]

          io_sets = iod_to_io_set_with_offset_time_cols(io_size, io_offset)



          # {
          #   input_set:  input_set,
          #   output_set: output_set,
          # }
        end
      end
    end
  end
end
