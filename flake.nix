{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    utils,
  }: let
    out = system: let
      pkgs = import nixpkgs {
        inherit system;
        config.allowBroken = true;
      };
      inherit (pkgs) mkShell poetry2nix python39;
      inherit (poetry2nix) mkPoetryApplication mkPoetryEnv;
      python = python39;
      overrides = pyfinal: pyprev: let
        inherit (pyprev) buildPythonPackage fetchPypi;
      in rec {};
      poetryArgs = {
        inherit python;
        projectDir = ./.;
        overrides = poetry2nix.overrides.withDefaults overrides;
      };
      poetryEnv = mkPoetryEnv poetryArgs;
      buildInputs = with pkgs; [
        # chromedriver
        # poetry
        # poetryEnv
        texlive.combined.scheme-full
        typst
        typst-lsp
        typstfmt
        typst-preview
        # wget
      ];
    in rec {
      devShell = mkShell rec {
        inherit buildInputs;
      };
    };
  in
    utils.lib.eachDefaultSystem out;
}
