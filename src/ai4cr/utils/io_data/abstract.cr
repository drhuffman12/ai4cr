require "./file_type"

module Ai4cr
  module Utils
    module IoData
      abstract class Abstract
        property file_path : String
        property raw = ""
        property iod = Array(Array(Float64)).new

        def initialize(@file_path : String, file_content_type : FileType)
          case file_content_type
          when FileType::Raw
            @raw = File.read(file_path)
            @iod = convert_raw_to_iod(@raw)
          when FileType::Iod
            contents = File.read(file_path)
            @iod = Array(Array(Float64)).from_json(contents)
            @raw = convert_iod_to_raw(@iod)
          end
        end

        def convert_raw_to_iod(raw) : Array(Array(Float64))
          raise "Must be implemented in subclass"
        end

        def convert_iod_to_raw(iod) : String
          raise "Must be implemented in subclass"
        end

        def save_raw(to_file_path : String)
          File.write(to_file_path, raw)
        end

        def save_iod(to_file_path : String)
          File.write(to_file_path, iod.to_json)
        end

        ####
        # Below is for RNN's or other AI algorythms requiring multiple time-column inputs or outputs

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

        # def iod_to_io_set_with_offset_time_cols(time_cols : Int32, offset : Int32)
        #   io_set = iod_to_io_set_with_offset(offset)
        #   time_col_indexes = Array.new(time_cols) { |i| i }

        #   input_i_max = io_set.size - 1
        #   output_i_max = io_set.size - 1

        #   ti_set_qty = io_set[:inputs].size - time_cols + 1
        #   ti_set_indexes = Array.new(ti_set_qty) { |i| i }

        #   input_set = ti_set_indexes.map do |si|
        #     time_col_indexes.map do |ti|
        #       io_set[:inputs][si..si+ti-1]
        #     end
        #   end

        #   output_set = ti_set_indexes.map do |si|
        #     time_col_indexes.map do |ti|
        #       io_set[:outputs][si..si+ti-1]
        #     end
        #   end

        #   {
        #     input_set: input_set,
        #     output_set: output_set
        #   }
        # end
      end
    end
  end
end
