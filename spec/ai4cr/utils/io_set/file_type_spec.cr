require "./../../../spectator_helper"

Spectator.describe Ai4cr::Utils::IoData::FileText do
  let(file_type_raw) { Ai4cr::Utils::IoData::FileType::Raw }
  let(file_type_io_formatted) { Ai4cr::Utils::IoData::FileType::Iod }

  describe "FileType" do
    context "Raw" do
      it "#to_s" do
        expect(file_type_raw.to_s).to eq("Raw")
      end

      it "#to_json" do
        old_stype = "0" # as of Crystal 0.36.0 [1e6f96aef] (2021-01-26)
        new_stype = "raw" # as of Crystal 1.0.0-dev [eef60c49b] (2021-02-25)
        expect([old_stype, new_stype]).to contain(file_type_raw.to_json)
      end
    end

    context "Iod" do
      it "#to_s" do
        expect(file_type_io_formatted.to_s).to eq("Iod")
      end

      it "#to_json" do
        old_stype = "1" # as of Crystal 0.36.0 [1e6f96aef] (2021-01-26)
        new_stype = "iod" # as of Crystal 1.0.0-dev [eef60c49b] (2021-02-25)
        expect([old_stype, new_stype]).to contain(file_type_io_formatted.to_json)
      end
    end
  end
end
