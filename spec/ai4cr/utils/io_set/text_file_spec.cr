require "./../../../spectator_helper"

Spectator.describe Ai4cr::Utils::IoData::FileText do
  let(temp_folder) { "spec/tmp" }

  before_each do
    Dir.mkdir_p(temp_folder)
  end

  let(file_path) { "./spec_bench/support/neural_network/data/bible_utf/eng-web_002_GEN_01_read.txt" }
  let(file_type_raw) { Ai4cr::Utils::IoData::FileType::Raw }
  let(file_type_iod) { Ai4cr::Utils::IoData::FileType::Iod }
  let(prefix_raw_qty) { 0 }
  let(prefix_raw_char) { " " }

  let(io_set_text_file) do
    Ai4cr::Utils::IoData::TextFile.new(
      file_path, file_type_raw,
      prefix_raw_qty, prefix_raw_char
    )
  end

  let(raw) { io_set_text_file.raw }
  let(iod) { io_set_text_file.iod }

  let(start_expected_10_chars) { "﻿The First" }
  let(end_expected_10_chars) { "sixth day. \n" }

  let(start_expected_3_iod) {
    [
      [
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      ],
      [
        0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      ],
      [
        0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      ],
    ]
  }
  let(end_expected_3_iod) {
    [
      [
        0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      ],
      [
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      ],
      [
        0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      ],
    ]
  }

  let(start_expected_3_chars) { "﻿Th" }
  let(end_expected_3_chars) { ". \n" }

  describe "#initialize" do
    describe "when given raw text data" do
      describe "assigns" do
        describe "raw, which" do
          it "starts as expected" do
            expect(raw[0..9]).to eq(start_expected_10_chars)
          end
          it "ends as expected" do
            expect(raw[-12..-1]).to eq(end_expected_10_chars)
          end
        end

        describe "iod, which" do
          it "starts as expected" do
            expect(iod[0..2]).to eq(start_expected_3_iod)
          end
          it "ends as expected" do
            expect(iod[-3..-1]).to eq(end_expected_3_iod)
          end
        end

        describe "raw, which when prefixed" do
          let(prefix_raw_qty) { 2 }
          let(prefix_char) { " " }
          let(prefix) { prefix_char + prefix_char }

          it "starts as expected" do
            expect(raw[0..9]).to eq((prefix + start_expected_10_chars)[0..9]) # [0..-3]
          end
          it "ends as expected" do
            expect(raw[-12..-1]).to eq(end_expected_10_chars)
          end
        end

        describe "iod, which when prefixed" do
          let(prefix_raw_qty) { 2 }
          let(prefix_bits) { Ai4cr::Utils::IoData::TextFile.convert_raw_to_iod(prefix_raw_char) }
          let(prefix) { prefix_bits + prefix_bits }

          it "starts as expected" do
            expect(iod[0..2]).to eq((prefix + start_expected_3_iod)[0..2])
          end
          it "ends as expected" do
            expect(iod[-3..-1]).to eq(end_expected_3_iod)
          end
        end
      end
    end

    describe "when given iod data" do
      let(temp_file_path) { "#{temp_folder}/temp_text_file_spec.#{Time.utc}.#{rand(1000)}.json" }

      let(iod_file) do
        io_set_text_file.save_iod(temp_file_path)
        Ai4cr::Utils::IoData::TextFile.new(temp_file_path, file_type_iod)
      end

      describe "assigns" do
        describe "raw, which" do
          it "starts as expected" do
            expect(raw[0..9]).to eq(start_expected_10_chars)
          end
          it "ends as expected" do
            expect(raw[-12..-1]).to eq(end_expected_10_chars)
          end
        end

        describe "iod, which" do
          it "starts as expected" do
            expect(iod[0..2]).to eq(start_expected_3_iod)
          end
          it "ends as expected" do
            expect(iod[-3..-1]).to eq(end_expected_3_iod)
          end
        end
      end
    end
  end

  describe "#convert_raw_to_iod" do
    context "when given the first 3 raw charaters" do
      it "correctly converts to iod" do
        converted = Ai4cr::Utils::IoData::TextFile.convert_raw_to_iod(start_expected_3_chars)
        expect(converted).to eq(start_expected_3_iod)
      end
    end

    context "when given the last 3 raw charaters" do
      it "correctly converts to iod" do
        converted = Ai4cr::Utils::IoData::TextFile.convert_raw_to_iod(end_expected_3_chars)
        expect(converted).to eq(end_expected_3_iod)
      end
    end
  end

  describe "#convert_iod_to_raw" do
    context "when given the first 3 iod charaters" do
      it "correctly converts to raw" do
        converted = Ai4cr::Utils::IoData::TextFile.convert_iod_to_raw(start_expected_3_iod)
        expect(converted).to eq(start_expected_3_chars)
      end
    end

    context "when given the last 3 iod charaters" do
      it "correctly converts to raw" do
        converted = Ai4cr::Utils::IoData::TextFile.convert_iod_to_raw(end_expected_3_iod)
        expect(converted).to eq(end_expected_3_chars)
      end
    end
  end

  describe "#bits_to_char" do
    let(bits) { end_expected_3_iod.first }
    let(char_expected) { end_expected_3_chars[0].to_s }

    it "correctly converts to char" do
      converted = Ai4cr::Utils::IoData::TextFile.bits_to_char(bits)
      expect(converted).to eq(char_expected)
    end
  end

  describe "#bytes_to_chars" do
    let(iod) { end_expected_3_iod }
    let(raw_expected) { end_expected_3_chars.split("") }

    it "correctly converts to raw text array" do
      converted = Ai4cr::Utils::IoData::TextFile.bytes_to_chars(iod)
      expect(converted).to eq(raw_expected)
    end
  end

  context "saving" do
    describe "#save_raw" do
      let(temp_file_path) { "#{temp_folder}/temp_text_file_spec.#{Time.utc.to_s("%Y-%m-%d_%H-%M-%S")}.#{rand(1000)}.txt" }

      it "correctly saves the raw contents" do
        io_set_text_file.save_raw(temp_file_path)

        contents = File.read(temp_file_path)

        expect(contents).to eq(io_set_text_file.raw)

        begin
          File.delete(temp_file_path)
        rescue
        end
      end
    end

    describe "#save_iod" do
      let(temp_file_path) { "#{temp_folder}/temp_text_file_spec.#{Time.utc}.#{rand(1000)}.json" }

      it "correctly saves the iod as json" do
        io_set_text_file.save_iod(temp_file_path)

        contents = File.read(temp_file_path)

        expect(contents).to eq(io_set_text_file.iod.to_json)

        begin
          File.delete(temp_file_path)
        rescue
        end
      end
    end
  end

  describe "#iod_to_io_set_with_offset" do
    let(offset) { 3 }
    let(ios) { io_set_text_file.iod_to_io_set_with_offset(offset) }

    context "returns expected data for" do
      context ":inputs" do
        context "start with" do
          it "raw" do
            snippet = ios[:inputs][0..2]
            expect(snippet).to be_a(Array(Array(Float64)))

            raw_text_expected = "﻿Th"

            snippet_as_text = Ai4cr::Utils::IoData::TextFile.convert_iod_to_raw(snippet)
            expect(snippet_as_text).to eq(raw_text_expected)
          end
        end
        context "end with" do
          it "raw" do
            snippet = ios[:inputs][-3..-1]
            expect(snippet).to be_a(Array(Array(Float64)))
            raw_text_expected = "day"

            snippet_as_text = Ai4cr::Utils::IoData::TextFile.convert_iod_to_raw(snippet)
            expect(snippet_as_text).to eq(raw_text_expected)
          end
        end
      end
      context ":outputs" do
        context "start with" do
          it "raw" do
            snippet = ios[:outputs][0..2]
            expect(snippet).to be_a(Array(Array(Float64)))

            raw_text_expected = "e F"

            snippet_as_text = Ai4cr::Utils::IoData::TextFile.convert_iod_to_raw(snippet)
            expect(snippet_as_text).to eq(raw_text_expected)
          end
        end
        context "end with" do
          it "raw" do
            snippet = ios[:outputs][-3..-1]
            expect(snippet).to be_a(Array(Array(Float64)))

            raw_text_expected = ". \n"

            snippet_as_text = Ai4cr::Utils::IoData::TextFile.convert_iod_to_raw(snippet)
            expect(snippet_as_text).to eq(raw_text_expected)
          end
        end
      end
    end
  end

  describe "#iod_to_io_set_with_offset_time_cols" do
    let(offset) { 3 }
    let(time_cols) { 4 }
    let(ios) { io_set_text_file.iod_to_io_set_with_offset_time_cols(time_cols, offset) }

    context "returns expected data for" do
      context ":input_set" do
        context "start with" do
          it "raw" do
            snippets = ios[:input_set][0..2]
            expect(snippets).to be_a(Array(Array(Array(Float64))))

            raw_text_array_expected = ["﻿The", "The ", "he F"]

            snippet_as_text_array = snippets.map { |snippet| Ai4cr::Utils::IoData::TextFile.convert_iod_to_raw(snippet) }
            expect(snippet_as_text_array).to eq(raw_text_array_expected)
          end
        end

        context "end with" do
          it "raw" do
            snippets = ios[:input_set][-3..-1]
            expect(snippets).to be_a(Array(Array(Array(Float64))))

            raw_text_array_expected = ["xth ", "th d", "h da"]

            snippet_as_text_array = snippets.map { |snippet| Ai4cr::Utils::IoData::TextFile.convert_iod_to_raw(snippet) }
            expect(snippet_as_text_array).to eq(raw_text_array_expected)
          end
        end
      end

      context ":output_set" do
        context "start with" do
          it "raw" do
            snippets = ios[:output_set][0..2]
            expect(snippets).to be_a(Array(Array(Array(Float64))))

            # raw_text_array_expected = ["﻿The", "The ", "he F"]
            raw_text_array_expected = ["e Fi", " Fir", "Firs"]

            snippet_as_text_array = snippets.map { |snippet| Ai4cr::Utils::IoData::TextFile.convert_iod_to_raw(snippet) }
            expect(snippet_as_text_array).to eq(raw_text_array_expected)
          end
        end
        context "end with" do
          it "raw" do
            snippets = ios[:output_set][-3..-1]
            expect(snippets).to be_a(Array(Array(Array(Float64))))

            # raw_text_array_expected = ["xth ", "th d", "h da"]
            raw_text_array_expected = [" day", "day.", "ay. "]

            snippet_as_text_array = snippets.map { |snippet| Ai4cr::Utils::IoData::TextFile.convert_iod_to_raw(snippet) }
            expect(snippet_as_text_array).to eq(raw_text_array_expected)
          end
        end
      end
    end
  end

  describe "(un)certainty)" do
    let(iod_guess_high_confidence) {
      [
        [
          0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        [
          0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        [
          0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
      ]
    }
    let(iod_guess_low_confidence) {
      [
        [
          -0.5, 1.5, -1.5, 1.5, -0.5, 1.5, -0.5, 0.35,
          0.15, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.25,
          -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.15,
          0.25, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.15,
        ],
        [
          -0.35, 0.5, 0.5, -0.5, 0.5, -1.5, 0.5, -0.25,
          0.5, -0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.35,
          -0.45, 0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.45,
          0.5, -0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.55,
        ],
        [
          -0.5, -1.5, -0.5, -1.5, -0.5, -0.5, -0.5, -0.65,
          0.55, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.75,
          -0.65, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.85,
          0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.95,
        ],
      ]
    }

    describe "#iod_uncertainty" do
      context "for a guess with a mix of only '0.0' and '1.0'" do
        it "returns '0.0'" do
          uncertainty = io_set_text_file.iod_uncertainty(iod_guess_high_confidence)
          expect(uncertainty).to eq(0.0)
        end
      end
      context "for a guess with a mix of values, not just '0.0' and '1.0'" do
        it "returns > '0.0'" do
          uncertainty = io_set_text_file.iod_uncertainty(iod_guess_low_confidence)
          expect(uncertainty).to be > 0.0
        end
      end
    end

    describe "#iod_certainty" do
      context "for a guess with a mix of only '0.0' and '1.0'" do
        it "returns '1.0'" do
          uncertainty = io_set_text_file.iod_certainty(iod_guess_high_confidence)
          puts_debug "uncertainty: #{uncertainty}"
          expect(uncertainty).to eq(1.0)
        end
      end
      context "for a guess with a mix of values, not just '0.0' and '1.0'" do
        it "returns < '1.0'" do
          certainty = io_set_text_file.iod_certainty(iod_guess_low_confidence)
          puts_debug "certainty: #{certainty}"
          expect(certainty).to be < 1.0
          expect(certainty).to eq(0.4843750000000001)
        end
      end
    end
  end

  context "handles misc Unicode char's" do
    let(file_path) { "./spec/support/unicode/examples.txt" }
    it "converts to bits and back" do
      bits = Ai4cr::Utils::IoData::TextFile.convert_raw_to_iod(io_set_text_file.raw)
      text = Ai4cr::Utils::IoData::TextFile.convert_iod_to_raw(bits)
      # puts "bits: #{bits}"
      prefix = (io_set_text_file.prefix_raw_char * io_set_text_file.prefix_raw_qty)
      expect(prefix + text).to eq(io_set_text_file.raw)
    end
  end
end
