var assert = require('assert')
  , fs = require('fs')
  , path = require('path')
  , bayes = require('../lib/naive_bayes')

describe('bayes() init', function () {
  it('valid options (falsey or with an object) do not raise Errors', function () {
    var validOptionsCases = [ undefined, {} ];

    validOptionsCases.forEach(function (validOptions) {
      var classifier = bayes(validOptions)
      assert.deepEqual(classifier.options, {})
    })
  })

  it('invalid options (truthy and not object) raise TypeError during init', function () {
    var invalidOptionsCases = [ null, 0, 'a', [] ];

    invalidOptionsCases.forEach(function (invalidOptions) {
      assert.throws(function () { bayes(invalidOptions) }, Error)
      // check that it's a TypeError
      assert.throws(function () { bayes(invalidOptions) }, TypeError)
    })
  })
})

describe('bayes using custom tokenizer', function () {
  it('uses custom tokenization function if one is provided in `options`.', function () {
    var splitOnChar = function (text) {
      return text.split('')
    }

    var classifier = bayes({ tokenizer: splitOnChar })

    classifier.learn('abcd', 'happy')

    // check classifier's state is as expected
    assert.equal(classifier.totalDocuments, 1)
    assert.equal(classifier.docCount.happy, 1)
    assert.deepEqual(classifier.vocabulary, { a: 1, b: 1, c: 1, d: 1 })
    assert.equal(classifier.vocabularySize, 4)
    assert.equal(classifier.wordCount.happy, 4)
    assert.equal(classifier.wordFrequencyCount.happy.a, 1)
    assert.equal(classifier.wordFrequencyCount.happy.b, 1)
    assert.equal(classifier.wordFrequencyCount.happy.c, 1)
    assert.equal(classifier.wordFrequencyCount.happy.d, 1)
    assert.deepEqual(classifier.categories, { happy: 1 })
  })
})

describe('bayes serializing/deserializing its state', function () {
  it('serializes/deserializes its state as JSON correctly.', function (done) {
      var classifier = bayes()

      classifier.learn('Fun times were had by all', 'positive')
      classifier.learn('sad dark rainy day in the cave', 'negative')

      var jsonRepr = classifier.toJson()

      // check serialized values
      var state = JSON.parse(jsonRepr)

      // ensure classifier's state values are all in the json representation
      bayes.STATE_KEYS.forEach(function (k) {
        assert.deepEqual(state[k], classifier[k])
      })

      var revivedClassifier = bayes.fromJson(jsonRepr)

      // ensure the revived classifier's state is same as original state
      bayes.STATE_KEYS.forEach(function (k) {
        assert.deepEqual(revivedClassifier[k], classifier[k])
      })

      done()
    })

  it('allows de-serializing an empty state', function (done) {
    var classifier = bayes();
    var jsonRepr = classifier.toJson()
    bayes.fromJson(jsonRepr);
    done();
  });

  it('fails on an missing fields', function (done) {
    var classifier = bayes();
    var jsonRepr = classifier.toJson();

    jsonRepr = JSON.parse(jsonRepr);
    delete jsonRepr.totalDocuments;
    jsonRepr = JSON.stringify(jsonRepr);

    try {
      bayes.fromJson(jsonRepr);
    } catch (e) {
      return done();
    }

    done(new Error('should have thrown'));
  })
})

describe('bayes .learn() correctness', function () {
  //sentiment analysis test
  it('categorizes correctly for `positive` and `negative` categories', function (done) {

    var classifier = bayes()

    //teach it positive phrases
    classifier.learn('amazing, awesome movie!! Yeah!!', 'positive')
    classifier.learn('Sweet, this is incredibly, amazing, perfect, great!!', 'positive')

    //teach it a negative phrase
    classifier.learn('terrible, shitty thing. Damn. Sucks!!', 'negative')

    //teach it a neutral phrase
    classifier.learn('I dont really know what to make of this.', 'neutral')

    //now test it to see that it correctly categorizes a new document
    classifier.categorize('awesome, cool, amazing!! Yay.')
      .then((data) => {
        assert.equal(data, 'positive')
        done()
      })
  })

  //topic analysis test
  it('categorizes correctly for `chinese` and `japanese` categories', function (done) {

    var classifier = bayes()

    //teach it how to identify the `chinese` category
    classifier.learn('Chinese Beijing Chinese', 'chinese')
    classifier.learn('Chinese Chinese Shanghai', 'chinese')
    classifier.learn('Chinese Macao', 'chinese')

    //teach it how to identify the `japanese` category
    classifier.learn('Tokyo Japan Chinese', 'japanese')

    //make sure it learned the `chinese` category correctly
    var chineseFrequencyCount = classifier.wordFrequencyCount.chinese

    assert.equal(chineseFrequencyCount['Chinese'], 5)
    assert.equal(chineseFrequencyCount['Beijing'], 1)
    assert.equal(chineseFrequencyCount['Shanghai'], 1)
    assert.equal(chineseFrequencyCount['Macao'], 1)

    //make sure it learned the `japanese` category correctly
    var japaneseFrequencyCount = classifier.wordFrequencyCount.japanese

    assert.equal(japaneseFrequencyCount['Tokyo'], 1)
    assert.equal(japaneseFrequencyCount['Japan'], 1)
    assert.equal(japaneseFrequencyCount['Chinese'], 1)

    //now test it to see that it correctly categorizes a new document
    assert.equal(classifier.categorize('Chinese Chinese Chinese Tokyo Japan'), 'chinese')

    done()
  })

  it('correctly tokenizes cyrlic characters', function (done) {
    var classifier = bayes()

    classifier.learn('Надежда за', 'a')
    classifier.learn('Надежда за обич еп.36 Тест', 'b')
    classifier.learn('Надежда за обич еп.36 Тест', 'b')

    var aFreqCount = classifier.wordFrequencyCount.a
    assert.equal(aFreqCount['Надежда'], 1)
    assert.equal(aFreqCount['за'], 1)

    var bFreqCount = classifier.wordFrequencyCount.b
    assert.equal(bFreqCount['Надежда'], 2)
    assert.equal(bFreqCount['за'], 2)
    assert.equal(bFreqCount['обич'], 2)
    assert.equal(bFreqCount['еп'], 2)
    assert.equal(bFreqCount['36'], 2)
    assert.equal(bFreqCount['Тест'], 2)

    done()
  })
})

describe('bayes .categorizeMultiple() correctness', function () {

  //sentiment analysis test
  it('categorizes correctly for `positive` and `negative` categories', function (done) {

    var classifier = bayes()

    //teach it positive phrases
    classifier.learn('amazing, awesome movie!! Yeah!!', 'positive')
    classifier.learn('Sweet, this is incredibly, amazing, perfect, great!!', 'positive')

    //teach it a negative phrase
    classifier.learn('terrible, shitty thing. Damn. Sucks!!', 'negative')

    //teach it a neutral phrase
    classifier.learn('I dont really know what to make of this.', 'neutral')

    //now test it to see that it correctly categorizes a new document
    const categories = 
      classifier
        .categorizeMultiple('awesome, cool, amazing!! Yay.', 2)
        .then(data => {
          //console.log(data);
          assert.equal(data.length, 2)
          assert.equal(data[0].category, 'positive')
          done()      
        })
    // assert.equal(categories[0], 'positive')
    //done()
  })
})

describe('bayes options correctness', function () {

  it('correctly behavior with no option specified', function (done) {

    var classifier = bayes()

    classifier.learn('just a word to be learned', 'whatever')
      .then((data) => {
        assert.equal(classifier.vocabularySize, 6)
        done()
      })
  })

  it('correctly consider minTokenSize option', function (done) {

    var classifier = bayes({
      minTokenSize: 2
    })

    classifier.learn('just a word to be learned', 'whatever')
      .then((data) => {
        // The word 'a' will not be considered (its length is less than 2), so vocabulary size must be 5
        assert.equal(classifier.vocabularySize, 5)
        done()
      })
  })

  it('correctly consider ignoredTokens option', function (done) {

    var classifier = bayes({
      ignoredTokens: ['a', 'to', 'be']
    })

    classifier.learn('just a word to be learned', 'whatever')
      .then((data) => {
        // The words 'a', 'to', and 'be' will not be considered, so vocabulary size must be 3
        assert.equal(classifier.vocabularySize, 3)
        done()
      })
  })

  it('correctly consider ignorePattern option', function (done) {

    var classifier = bayes({
      ignorePattern: /[^A-Za-z ]+/g
    })

    classifier.learn('just a word to be learned 123 (*)', 'whatever')
      .then((data) => {
        // The characters '123', and '(*)' will not be considered, so vocabulary size must be 6
        assert.equal(classifier.vocabularySize, 6)
        assert.equal(classifier.vocabulary['123'], undefined)
        done()
      })
  })
})