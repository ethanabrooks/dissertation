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
      buildInputs = with pkgs; [
        typst
        typst-lsp
        typstfmt
        typst-preview
      ];
    in rec {
      devShell = pkgs.mkShell rec {
        inherit buildInputs;
      };
    };
  in
    utils.lib.eachDefaultSystem out;
}
