class SafeCounter
  # Thanks to https://itnext.io/comparing-crystals-concurrency-with-that-of-go-part-ii-89049701b1a5
  #
  # Example Usage:
  # ```
  # c = SafeCounter.new
  # (1..1000).each {
  #   spawn c.inc("someKey")
  # }
  #
  # sleep 1.second
  # puts c.value("someKey")
  # ```

  def initialize
    @v = Hash(String, Int32).new
    @mux = Mutex.new
  end

  def inc(key : String)
    @mux.lock
    value = @v.has_key?(key) ? @v[key] + 1 : 1
    @v[key] = value
  ensure
    @mux.unlock
    value
  end

  def value(key : String)
    @mux.lock
    @v[key]
  ensure
    @mux.unlock
  end
end
