require "./../../../spectator_helper"

Spectator.describe Ai4cr::Utils::IoData::FileText do
  let(file_type_raw) { Ai4cr::Utils::IoData::FileType::Raw }
  let(file_type_iod) { Ai4cr::Utils::IoData::FileType::Iod }

  describe "FileType" do
    context "Raw" do
      it "#to_s" do
        expect(file_type_raw.to_s).to eq("Raw")
      end

      pending "#to_json" do
        # old_type = "0"   # as of Crystal 0.36.0 [1e6f96aef] (2021-01-26)
        new_type = "raw" # as of Crystal 1.0.0-dev [eef60c49b] (2021-02-25)

        expect(file_type_raw.to_json).to eq(new_type)
      end
    end

    context "Iod" do
      it "#to_s" do
        expect(file_type_iod.to_s).to eq("Iod")
      end

      pending "#to_json" do
        # old_type = "1"   # as of Crystal 0.36.0 [1e6f96aef] (2021-01-26)
        new_type = "iod" # as of Crystal 1.0.0-dev [eef60c49b] (2021-02-25)

        expect(file_type_iod.to_json).to eq(new_type)
      end
    end
  end
end
