require "benchmark"
# require "./../../../../src/ai4cr.cr"
require "../../../src/ai4cr.cr"

FILE_PATH = "./spec_bench/support/neural_network/data/bible_utf/eng-web_002_GEN_01_read.txt"

Benchmark.ips do |x|
  x.report "String IO" do
    File.open FILE_PATH do |file|
      # Built the bits out as a string, otherwise may run into Integer size limits pretty quickly.
      String.build do |io|
        file.each_byte do |byte|
          byte.to_s io, base: 2
        end
      end
    end
  end

  x.report "2D Array" do
    bits = Array(Array(Int32)).new
    File.read(FILE_PATH).each_char do |char|
      bytes = char.ord # 4 Bytes, so 32 bits
      bits << (0..31).to_a.map { |i| bytes.bit(i) }
      bits
    end
    bits
  end

  x.report "2D Array utils" do
    Ai4cr::Utils::IoData::TextFile.new(FILE_PATH, Ai4cr::Utils::IoData::FileType::Raw).iod
  end
end

# i = 0
# puts
# bits = Array(Array(Float64)).new(32)
# File.open FILE_PATH do |file|
#   # Built the bits out as a string, otherwise may run into Integer size limits pretty quickly.
#   String.build do |io|
#     file.each_byte do |byte|
#       # byte.to_s io, base: 2
#       bits << byte.to_s io, base: 2
#       # puts(byte.to_s(io, base: 2)) if i < 10
#       # i += 1
#     end
#   end
# end

# bits = File.open FILE_PATH do |file|
#   # Built the bits out as a string, otherwise may run into Integer size limits pretty quickly.
#   String.build do |io|
#     file.each_byte do |byte|
#       (byte.to_s(io, base: 2)) # .&split("")
#     end
#   end
# end
# puts bits

# bits = File.open FILE_PATH do |file|
#   # Built the bits out as a string, otherwise may run into Integer size limits pretty quickly.
#   String.build do |io|
#     file.each_char.map do |char|
#       (byte.to_s(io, base: 2)) # .&split("")
#     end
#   end
# end
# puts bits
