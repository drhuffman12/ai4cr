require "benchmark"
require "../../../src/ai4cr.cr"

FILE_PATH = "./spec_bench/support/neural_network/data/bible_utf/eng-web_002_GEN_01_read.txt"

Benchmark.ips do |x|
  x.report "String IO" do
    File.open FILE_PATH do |file|
      # Built the bits out as a string, otherwise may run into Integer size limits pretty quickly.
      String.build do |io|
        file.each_byte do |byte|
          byte.to_s(io, base: 2)
        end
      end
    end # => String; wrong type!
  end

  x.report "String IO2" do
    File.open FILE_PATH do |file|
      String.build do |io|
        file.each_char do |char|
          char.to_s(io)
        end
      end
    end # => String; wrong type!
  end

  x.report "2D Array" do
    bits = Array(Array(Int32)).new
    File.read(FILE_PATH).each_char do |char|
      bytes = char.ord # 4 Bytes, so 32 bits
      bits << (0..31).to_a.map { |i| bytes.bit(i) }
      bits
    end
    bits # => Array(Array(Int32)); close enough, but slow!
  end

  x.report "2D Array utils" do
    # The winner! :)
    Ai4cr::Utils::IoData::TextFileIodBits.new(FILE_PATH, Ai4cr::Utils::IoData::FileType::Raw).iod
  end
end

# $ crystal spec spec_bench/ai4cr/io_data/text_file_spec.cr --release
#      String IO   9.25k (108.10µs) (± 7.09%)  73.1kB/op   2.55× slower
#     String IO2  23.57k ( 42.42µs) (± 6.11%)  21.1kB/op        fastest
#       2D Array 416.11  (  2.40ms) (± 8.92%)  2.52MB/op  56.65× slower
# 2D Array utils   1.02k (983.61µs) (± 7.09%)  1.18MB/op  23.19× slower
