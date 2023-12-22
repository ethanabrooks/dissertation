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
      };
      my-typst-preview = import ./pkgs/typst-preview/package.nix {
        inherit (pkgs) lib rustPlatform fetchFromGitHub mkYarnPackage fetchYarnDeps pkg-config libgit2 openssl zlib stdenv darwin;
      };

      buildInputs = with pkgs; [
        my-typst-preview
        pdf2svg
        typst
        typstfmt
        typst-lsp
        wget
      ];
    in rec {
      devShell = pkgs.mkShell rec {
        inherit buildInputs;
      };
    };
  in
    utils.lib.eachDefaultSystem out;
}
