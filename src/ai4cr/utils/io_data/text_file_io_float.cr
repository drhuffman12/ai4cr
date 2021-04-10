require "./abstract"

module Ai4cr
  module Utils
    module IoData
      class TextFileIodFloat < Ai4cr::Utils::IoData::Abstract
        # TODO: refactor (or add/rename methods) so we can go from one-char per time-column (in bits per tc's inputs)
        #   to multiple char's per time col (several per tc's inputs)
        #   REMINDER: Must consider Ai4cr::NeuralNetwork::Rnn::RnnSimpleConcerns::PropsAndInits::INPUT_SIZE_MIN and OUTPUT_SIZE_MIN

        UTF_MAX_AS_FLOAT        = Char::MAX_CODEPOINT.to_f
        UTF_MAX_CHAR_STR        = Char::MAX_CODEPOINT.chr.to_s
        CHARS_AT_A_TIME_DEFAULT = 4
        INPUT_SIZE_MIN          = Ai4cr::NeuralNetwork::Rnn::RnnSimpleConcerns::PropsAndInits::INPUT_SIZE_MIN

        def self.convert_raw_to_iod(raw, chars_at_a_time = CHARS_AT_A_TIME_DEFAULT) : Array(Array(Float64))
          chars_as_floats = Array(Array(Float64)).new(raw.size)
          raw.each_char { |char| chars_as_floats << char_to_charaf(char) }
          chars_as_floats
        end

        def self.resize_chars_at_a_time(raw, chars_at_a_time)
          chars_at_a_time = INPUT_SIZE_MIN if chars_at_a_time < INPUT_SIZE_MIN
          chars_at_a_time = raw.size if chars_at_a_time > raw.size
          chars_at_a_time
        end

        def self.char_to_charaf(char)
          [1.0 * char.ord / Char::MAX_CODEPOINT]
        end

        def self.charaf_to_char(charaf)
          char_val = (charaf.first * Char::MAX_CODEPOINT)
          case
          when char_val.nan? || char_val.infinite?
            Char::REPLACEMENT
          when char_val.round < 0
            0.chr.to_s
          when char_val.round > UTF_MAX_AS_FLOAT
            UTF_MAX_CHAR_STR
          else
            char_val.round.to_i.chr.to_s
          end
        end

        def self.convert_iod_to_raw(iod, chars_at_a_time = CHARS_AT_A_TIME_DEFAULT) : String
          iod.join { |charaf| charaf_to_char(charaf) }
        end
      end
    end
  end
end
