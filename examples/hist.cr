list = Array(Int32).new
hist = Hash(Int32, Int32).new(0)
perc = Hash(Int32, Float64).new(0.0)
all_hists = Array(Hash(Int32, Int32)).new

p! list

loop_size = 50
tc_size = 10

loop_size.times do
  p! i = rand(tc_size)
  list << i
end

p! list
p! list.size

tc_size.times do |i|
  hist[i + 1] = 0
  perc[i + 1] = 0.0
end
list.each do |i|
  hist[i + 1] += 1
end
tc_size.times do |i|
  perc[i + 1] = 100.0 * hist[i + 1] / tc_size
end

tc_size.times do |i|
  puts "hist[#{i + 1}] : #{hist[i + 1]}"
  puts "perc[#{i + 1}] : #{perc[i + 1]}"
end

all_hists << hist

all_hists = all_hists[-100..-1] if all_hists.size > 100

p! hist
p! perc
p! all_hists
