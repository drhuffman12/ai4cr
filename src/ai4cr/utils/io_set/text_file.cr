require "./abstract"

module Ai4cr
  module Utils
    module IoSet
      class TextFile < Ai4cr::Utils::IoSet::Abstract
        BIT_32_INDEXES = (0..31).to_a

        # include Ai4cr::Utils::IoSet::Common

        ################

        # def self.text_file_to_fios(raw)
        def convert_raw_to_ios(raw)
          chars_as_bytes_of_bits = Array(Array(Float64)).new(raw.size)
          raw.each_char { |char| chars_as_bytes_of_bits << char_to_bits(char) }
          chars_as_bytes_of_bits
        end

        def char_to_bits(char)
          bytes = char.ord
          BIT_32_INDEXES.map { |i| bytes.bit(i) * 1.0 }
        end

        ################

        def bits_to_char(bits)
          bytes = BIT_32_INDEXES.map { |i| bits[i] * (2.0**i) }.sum.to_i
          bytes.chr
        end

        def bytes_to_char(bytes)
          bytes.map { |bits| bits_to_char(bits) }
        end

        def chars_to_text(chars)
          chars.join
        end

        def convert_ios_to_raw(ios)
          "TBD"
        end

        ################
        # def self.offset_io_pairs_to_text_pairs(offset_io_pairs)
        #   puts offset_io_pairs
        #   # offset_io_pairs.map { |a| puts a }
        #   "test"
        # end

        # def self.fios_to_offset_io_sets(fios : Array(Array(Float64)), ti_size : Int32, ti_offset : Int32) : Array(Array(NamedTuple(inputs: Array(Float64), outputs: Array(Float64))))
        #   # io_pairs = Array(Array(Float64))
        #   raise ArgumentError.new("Bad Time Index Size") if ti_size < 1
        #   raise ArgumentError.new("Bad Time Index Offset") if ti_offset < 1

        #   input_set = Array(Array(Float64)).new
        #   output_set = Array(Array(Float64)).new

        #   fios_i_max = fios.size - 1
        #   in_i_last = fios_i_max - ti_size - ti_offset - 1
        #   in_ti_indexes = (0..in_i_last).to_a
        #   ti_indexes = (0..ti_size - 1).to_a

        #   # inputs = Array(Array(Float64)).new
        #   # outputs = Array(Array(Float64)).new
        #   # in_ti_indexes.map do |in_i_first|
        #   #   out_i_first = in_i_first + ti_offset

        #   #   input_set_sub = Array(Float64).new
        #   #   output_set_sub = Array(Float64).new

        #   #   ti_indexes.map_with_index { |ti_i|
        #   #     in_i = in_i_first + ti_i
        #   #     out_i = out_i_first + ti_i

        #   #     input_set_sub << fios[in_i]
        #   #     putput_set_sub << fios[out_i]
        #   #     {
        #   #       inputs: fios[in_i],
        #   #       outputs: fios[out_i]
        #   #     }
        #   #     # inputs << fios[in_i]
        #   #     # outputs << fios[out_i]
        #   #   }

        #   input_set = in_ti_indexes.map do |in_i_first|
        #     ti_indexes.map_with_index { |ti_i|
        #       in_i = in_i_first + ti_i
        #       inputs: fios[in_i]
        #     }
        #   end

        #   output_set = in_ti_indexes.map do |in_i_first|
        #     out_i_first = in_i_first + ti_offset
        #     ti_indexes.map_with_index { |ti_i|
        #       out_i = out_i_first + ti_i
        #       outputs: fios[out_i]
        #     }
        #   end
        #   { input_set: input_set, output_set: output_set }
        # end
      end
    end
  end
end
