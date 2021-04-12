require "json"
# struct Float64
#   include ::JSON::Serializable
# end

# Test what max values can be to/from-jsonified
(0..309).each do |exp|
  puts "*"*20
  p! exp
  f = 0.0
  p!(f = 10.0**(-exp))
  p! f
  j = f.to_json
  p! j
  f2 = Float64.from_json(j)
  p! f2
end
