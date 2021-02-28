require "./../../../spectator_helper"

Spectator.describe Ai4cr::Utils::IoSet::FileText do
  let(file_path) { "./spec_bench/support/neural_network/data/eng-web_002_GEN_01_read.txt" }
  # let(file_contents) { File.read(file_path) }
  let(file_to_text) { Ai4cr::Utils::IoSet::TextFile.file_to_text(file_path) }

  let(fios_from_file) {
    Ai4cr::Utils::IoSet::TextFile.text_file_to_fios(file_path)
  }
  let(fios_to_text) {
  }

  describe ".file_to_text" do
    it "starts as expected" do
      text_start_expected = "﻿The First"
      # [0..9]
      # expect(file_to_text[0..7]).to start_with(text_start_expected)
      expect(file_to_text[0..9]).to eq(text_start_expected)
    end
    it "end as expected" do
      text_end_expected = "sixth day. \n"
      # [-10..-1]
      # expect(file_to_text).to end_with(text_end_expected)
      expect(file_to_text[-12..-1]).to eq(text_end_expected)
    end
  end

  describe ".text_file_to_fios" do
  end
  # describe ".char_to_bits" do
  #   let(char) { 'C' }
  #   let(bits_expected) {
  #     [
  #       1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
  #       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  #       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  #       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
  #     ]
  #   }
  #   it "returns expected data" do
  #     bits = Ai4cr::Utils::IoSet::TextFile.char_to_bits(char)
  #     expect(bits).to eq(bits_expected)
  #   end
  # end

  # describe ".text_file_to_fios" do
  #   let(chars_from_file) {
  #     chars = Array(Char).new
  #     file_contents.each_char { |c| chars << c }
  #     chars
  #   }
  #   let(bytes_from_file) {
  #     bytes = Array(Int32).new
  #     file_contents.each_char { |char| bytes << char.ord }
  #     bytes
  #   }
  #   let(bit32_indexes) { Ai4cr::Utils::IoSet::TextFile::BIT_32_INDEXES }

  #   it "file content size is as expected" do
  #     expect(file_contents.size).to eq(3955)
  #   end
  #   it "beginning of file content are as expected" do
  #     expect(file_contents[0..39]).to eq("﻿The First Book of Moses, Commonly Calle")
  #   end
  #   it "beginning characters are as expected" do
  #     chars_expected = [
  #       '﻿', 'T', 'h', 'e', ' ',
  #       'F', 'i', 'r', 's', 't',
  #       ' ', 'B', 'o', 'o', 'k',
  #       ' ', 'o', 'f', ' ', 'M',
  #       'o', 's', 'e', 's', ',',
  #       ' ', 'C', 'o', 'm', 'm',
  #       'o', 'n', 'l', 'y', ' ',
  #       'C', 'a', 'l', 'l', 'e'
  #     ]
  #     expect(chars_from_file[0..39]).to eq(chars_expected)
  #   end
  #   it "beginning bytes are as expected" do
  #     chars_expected = [
  #       65279, 84, 104, 101, 32,
  #       70, 105, 114, 115, 116,
  #       32, 66, 111, 111, 107,
  #       32, 111, 102, 32, 77,
  #       111, 115, 101, 115, 44,
  #       32, 67, 111, 109, 109,
  #       111, 110, 108, 121, 32,
  #       67, 97, 108, 108, 101
  #     ]
  #     expect(bytes_from_file[0..39]).to eq(chars_expected)
  #   end

  #   it "bit32_indexes are as expected" do
  #     bit32_indexes_expected = [
  #       0, 1, 2, 3, 4,
  #       5, 6, 7, 8, 9,
  #       10, 11, 12, 13, 14,
  #       15, 16, 17, 18, 19,
  #       20, 21, 22, 23, 24,
  #       25, 26, 27, 28, 29,
  #       30, 31
  #     ]
  #     expect(bit32_indexes).to eq(bit32_indexes_expected)
  #   end

  #   it "beginning bits are as expected" do
  #     fios_expected = [
  #       [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  #       [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  #       [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  #       [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  #       [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  #       [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  #       [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  #       [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  #     ]
  #     expect(fios_from_file[0..7]).to eq(fios_expected)
  #   end
  # end

  # describe ".char_to_bits" do
  #   let(char) { 'C' }
  #   let(bits_expected) {
  #     [
  #       1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
  #       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  #       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  #       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
  #     ]
  #   }
  #   it "returns expected data" do
  #     bits = Ai4cr::Utils::IoSet::TextFile.char_to_bits(char)
  #     expect(bits).to eq(bits_expected)
  #   end
  # end

  # describe ".bits_to_char" do
  #   let(char_expected) { 'C' }
  #   let(bits) {
  #     [
  #       1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
  #       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  #       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  #       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
  #     ]
  #   }
  #   it "returns expected data" do
  #     char = Ai4cr::Utils::IoSet::TextFile.bits_to_char(bits)
  #     expect(char).to eq(char_expected)
  #   end
  # end

  # describe ".bits_to_char" do
  #   let(char_expected) { 'C' }
  #   let(bits) {
  #     [
  #       1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
  #       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  #       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  #       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
  #     ]
  #   }
  #   it "returns expected data" do
  #     char = Ai4cr::Utils::IoSet::TextFile.bits_to_char(bits)
  #     expect(char).to eq(char_expected)
  #   end
  # end

  # describe ".fios_to_offset_io_sets" do
  #   let(ti_size) { 10 }
  #   let(ti_offset) { 1 }
  #   let(io_sets) {
  #     Ai4cr::Utils::IoSet::TextFile.fios_to_offset_io_sets(fios_from_file, ti_size, ti_offset)
  #   }
  #   let(io_set_count_expected) { file_contents.size - ti_size - ti_offset - 1 }
  #   # let(io_sets_first_expected) {
  #   #   {
  #   #     inputs: [
  #   #       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
  #   #       0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
  #   #       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  #   #       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
  #   #     ],
  #   #     outputs: [
  #   #       0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
  #   #       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  #   #       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  #   #       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
  #   #     ]
  #   #   }
  #   # }
  #   # let(io_sets_last_expected) {
  #   #   {
  #   #     inputs: [
  #   #       1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0,
  #   #       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  #   #       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  #   #       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
  #   #     ],
  #   #     outputs: [
  #   #       0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
  #   #       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  #   #       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  #   #       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
  #   #     ]}
  #   # }
  #   let(text_pairs_last_expected) {
  #     {inputs: 'Y', outputs: 'Z'}
  #   }
  #   it "io pairs's count is as expected" do
  #     expect(io_sets.size).to eq(io_set_count_expected)
  #   end
  #   # it "io pair's '.first' count of is as expected" do
  #   #   expect(io_sets.first.size).to eq(ti_size)
  #   # end
  #   # it "io pair's '.first.first' is a NamedTuple of inputs and outputs" do
  #   #   keys_expected = {:inputs, :outputs}
  #   #   expect(io_sets.first.first.keys).to eq(keys_expected)
  #   # end
  #   # it "first io pair is as expected" do
  #   #   expect(io_sets.first.first).to eq(io_sets_first_expected)

  #   #   input_char = Ai4cr::Utils::IoSet::TextFile.bits_to_char(io_sets.first.first[:inputs])
  #   #   expect(input_char).to eq('﻿')

  #   #   output_char = Ai4cr::Utils::IoSet::TextFile.bits_to_char(io_sets.first.first[:outputs])
  #   #   expect(output_char).to eq('T')

  #   #   text = Ai4cr::Utils::IoSet::TextFile.offset_io_sets_to_text_pairs(io_sets)
  #   #   expect(text).to eq("tbd")
  #   # end
  #   # it "last io pair is as expected" do
  #   #   expect(io_sets.last.last).to eq(io_sets_last_expected)

  #   #   input_char = Ai4cr::Utils::IoSet::TextFile.bits_to_char(io_sets.last.last[:inputs])
  #   #   expect(input_char).to eq('l')

  #   #   output_char = Ai4cr::Utils::IoSet::TextFile.bits_to_char(io_sets.last.last[:outputs])
  #   #   expect(output_char).to eq('e')
  #   # end
  # end
end
