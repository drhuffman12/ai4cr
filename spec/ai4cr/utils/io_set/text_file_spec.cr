require "./../../../spectator_helper"

Spectator.describe Ai4cr::Utils::IoSet::FileText do
  let(file_type_raw) { Ai4cr::Utils::IoSet::FileType::Raw }
  let(file_type_io_formatted) { Ai4cr::Utils::IoSet::FileType::IoFormatted }

  describe "#initialize" do
    let(file_path) { "./spec_bench/support/neural_network/data/eng-web_002_GEN_01_read.txt" }
    let(io_set_text_file) { Ai4cr::Utils::IoSet::TextFile.new(file_path, file_type_raw) }

    let(raw) { io_set_text_file.raw }
    let(ios) { io_set_text_file.ios }

    describe "raw" do
      it "starts as expected" do
        text_start_expected = "﻿The First"
        # [0..9]
        # expect(file_to_text[0..7]).to start_with(text_start_expected)
        expect(raw[0..9]).to eq(text_start_expected)
      end
      it "end as expected" do
        text_end_expected = "sixth day. \n"
        # [-10..-1]
        # expect(file_to_text).to end_with(text_end_expected)
        expect(raw[-12..-1]).to eq(text_end_expected)
      end
    end
  end

end
