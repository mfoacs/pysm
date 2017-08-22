#!/bin/zsh

zipline ingest -b quantopian-quandl
zipline clean -bundle -b quantopian-quandl -k 1

