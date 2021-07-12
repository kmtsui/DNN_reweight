#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>

#include <cppflow/ops.h>
#include <cppflow/model.h>

#include <TROOT.h>
#include <TApplication.h>
#include <TStyle.h>
#include <TFile.h>
#include <TTree.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TChain.h>
#include <TBranch.h>
#include <TH1.h>
#include <TH2.h>
#include <TH3.h>
#include <TMath.h>
#include <TGraph.h>
#include <TVector3.h>

int main(int argc, char **argv) {

    char * modelDir=NULL;
    char * inputFileName=NULL;
    char * outputFileName=NULL;

    double xsec_norm = 1.;

    char c;
    while( (c = getopt(argc,argv,"m:i:o:n:h")) != -1 ){//input in c the argument (-f etc...) and in optarg the next argument. When the above test becomes -1, it means it fails to find a new argument.
        switch(c){
            case 'm':
                modelDir = optarg;
                break;
            case 'i':
                inputFileName = optarg;
                break;
            case 'o':
                outputFileName = optarg;
                break;
            case 'n':
                xsec_norm = std::stod(optarg);
                break;
            case 'h':
                std::cout << "USAGE: "
                        << argv[0] << "\nOPTIONS:\n"
                        << "-m : Model directory\n"
                        << "-i : Input file\n"
                        << "-o : Output weight file\n"
                        << "-n : Cross-section normalization\n"
                        << "-h : This help message\n";
            default:
                return 0;
        }
    }

    if (modelDir==NULL){
        std::cout << "Error, no input model!" << std::endl;
        return -1;
    }

    if (inputFileName==NULL){
        std::cout << "Error, no input file!" << std::endl;
        return -1;
    }

    if (outputFileName==NULL){
        outputFileName = (char*)"weight.root";
        std::cout << "No output file name defined. Setting output to weight.root" << std::endl;
    }

    TFile* fIn = TFile::Open(inputFileName);
    TTree* t = (TTree*)fIn->Get("RES");
    Float_t Enu, Q2, p_mu, costh_mu, p_pi, costh_pi, costh_mu_pi, W_had, gamma_had;
    Int_t mode;
    Double_t norm;
    t->SetBranchAddress("Enu",&Enu);
    t->SetBranchAddress("Q2",&Q2);
    t->SetBranchAddress("p_mu",&p_mu);
    t->SetBranchAddress("costh_mu",&costh_mu);
    t->SetBranchAddress("p_pi",&p_pi);
    t->SetBranchAddress("costh_pi",&costh_pi);
    t->SetBranchAddress("costh_mu_pi",&costh_mu_pi);
    t->SetBranchAddress("W_had",&W_had);
    t->SetBranchAddress("gamma_had",&gamma_had);
    t->SetBranchAddress("mode",&mode);
    t->SetBranchAddress("norm",&norm);

    TFile* fOut = new TFile(outputFileName,"RECREATE");
    TTree* t_reweight = new TTree("DNN_reweight","DNN_reweight");
    Float_t reweight;
    t_reweight->Branch("reweight",&reweight);

    const int nEntries = t->GetEntries();

    // Load the pre-trained model
    cppflow::model model(modelDir);
    // Vector for storing DNN classification variables
    std::vector<float> input_list(6);
    // Default method name for tensorflow I/O
    std::string input_string("serving_default_input_1:0");
    std::string output_string("StatefulPartitionedCall:0");

    std::cout << "--- Processing: " << nEntries << " events" << std::endl;

    const int nHists = 9;
    TH1D* hist_nominal[nHists];
    TH1D* hist_reweight[nHists];

    const std::string histName[nHists] = {
        "Enu", "Q2", "p_mu", "costh_mu", "p_pi", "costh_pi", "costh_mu_pi", "W_had", "gamma_had"
    };
    const double histLow[nHists] = {
        0,      0,    0,      -1,         0,    -1,           -1,            1.08,    1 
    };
    const double histHigh[nHists] = {
        6,      2,    6,       1,       1.5,     1,            1,            1.60,   2
    };
    const std::string histXaxis[nHists] = {
        "Enu (GeV)", "Q2 (GeV^{2})", "p_mu (GeV)", "costh_mu", "p_pi (GeV)", "costh_pi", "costh_mu_pi", "W_had (GeV)", "gamma_had"
    };

    for (int i=0;i<nHists;i++)
    {
        hist_nominal[i] =  new TH1D(Form("hist_nominal_%s",histName[i].c_str()),"",40,histLow[i],histHigh[i]);
        hist_reweight[i] = new TH1D(Form("hist_reweight_%s",histName[i].c_str()),"",40,histLow[i],histHigh[i]);
    }

    int weight_count = 0;
    double weight_sum = 0;
    for (int i=0;i<nEntries;i++)
    {
        if (i%1000 == 0) std::cout << "--- ... Processing event: " << i << std::endl;

        t->GetEntry(i);
        input_list[0] = Enu;
        input_list[1] = Q2;
        input_list[2] = p_mu;
        input_list[3] = costh_mu;
        input_list[4] = W_had;
        input_list[5] = gamma_had;

        auto input = cppflow::tensor ( input_list ,{1,6} );

        auto output = model(input, input_string, output_string);

        auto values = output.get_data<float>();

        reweight = values[0];
        reweight = reweight/(1-reweight);
        t_reweight->Fill();

        if (mode!=11) continue;

        weight_count++;
        weight_sum+=reweight;

        hist_nominal[0]->Fill(Enu, norm);
        hist_reweight[0]->Fill(Enu, norm*reweight*xsec_norm);
        hist_nominal[1]->Fill(Q2, norm);
        hist_reweight[1]->Fill(Q2, norm*reweight*xsec_norm);
        hist_nominal[2]->Fill(p_mu, norm);
        hist_reweight[2]->Fill(p_mu, norm*reweight*xsec_norm);
        hist_nominal[3]->Fill(costh_mu, norm);
        hist_reweight[3]->Fill(costh_mu, norm*reweight*xsec_norm);
        hist_nominal[4]->Fill(p_pi, norm);
        hist_reweight[4]->Fill(p_pi, norm*reweight*xsec_norm);
        hist_nominal[5]->Fill(costh_pi, norm);
        hist_reweight[5]->Fill(costh_pi, norm*reweight*xsec_norm);
        hist_nominal[6]->Fill(costh_mu_pi, norm);
        hist_reweight[6]->Fill(costh_mu_pi, norm*reweight*xsec_norm);
        hist_nominal[7]->Fill(W_had, norm);
        hist_reweight[7]->Fill(W_had, norm*reweight*xsec_norm);
        hist_nominal[8]->Fill(gamma_had, norm);
        hist_reweight[8]->Fill(gamma_had, norm*reweight*xsec_norm);
    }

    gStyle->SetOptStat(0);
    TCanvas* c1 = new TCanvas();
    TLegend* legend = new TLegend(0.6,0.6,0.9,0.9);
    legend->AddEntry(hist_nominal[0],"NuWro Eb = 0","l");
    legend->AddEntry(hist_reweight[0],"NuWro Eb = 27 MeV","l");
    legend->SetBorderSize(0);
    legend->SetFillStyle(0);
    for (int i=0;i<nHists;i++)
    {
        hist_nominal[i]->GetXaxis()->SetTitle(histXaxis[i].c_str());
        hist_nominal[i]->SetLineWidth(3);
        hist_reweight[i]->SetLineWidth(3);
        hist_nominal[i]->SetLineColor(kBlue);
        hist_reweight[i]->SetLineColor(kRed);
        hist_nominal[i]->Draw("hist");
        hist_reweight[i]->Draw("hist same");
        legend->Draw("same");
        c1->SaveAs(Form("hist_%s.pdf",histName[i].c_str()));
    }

    fOut->cd();
    t_reweight->Write();
    for (int i=0;i<nHists;i++)
    {
        hist_nominal[i]->Write();
        hist_reweight[i]->Write();
    }
    fOut->Close();

    fIn->Close();

    return 0;
}
