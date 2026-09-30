"use strict";

// Página-história do desafio AutoCred. Todos os números vêm de dados.json, e todos os textos
// de textos.json (PT e EN). Nada de bibliotecas: os gráficos são SVG montados aqui.

const LOCALES = { pt: "pt-BR", en: "en-US" };
const CHAVE_IDIOMA = "autocred.idioma";
const CENARIOS = ["brando", "base", "severo"];
const SVG_NS = "http://www.w3.org/2000/svg";

const estado = { idioma: "pt", dados: null, textos: null, sim: null, largura: 0 };

// ---------- idioma e textos ----------

function idiomaInicial() {
  const daUrl = new URLSearchParams(location.search).get("lang");
  if (daUrl && daUrl in LOCALES) return daUrl;
  try {
    const salvo = localStorage.getItem(CHAVE_IDIOMA);
    if (salvo && salvo in LOCALES) return salvo;
  } catch (erro) {
    // navegador sem armazenamento: segue com o padrão
  }
  return "pt";
}

function salvarIdioma(idioma) {
  try {
    localStorage.setItem(CHAVE_IDIOMA, idioma);
  } catch (erro) {
    // sem armazenamento, a escolha vale só para esta visita
  }
}

function t(chave, vars = {}) {
  const texto = estado.textos[estado.idioma][chave] ?? chave;
  return texto.replace(/\{(\w+)\}/g, (marca, nome) => (nome in vars ? vars[nome] : marca));
}

function aplicarTextos() {
  document.documentElement.lang = LOCALES[estado.idioma];
  document.title = t("meta.titulo");
  document.querySelectorAll("[data-t]").forEach((el) => { el.textContent = t(el.dataset.t); });
  document.querySelectorAll("[data-t-html]").forEach((el) => { el.innerHTML = t(el.dataset.tHtml); });
  $("idioma").setAttribute("aria-label", t("nav.idioma_rotulo"));
  $("idioma").title = t("nav.idioma_rotulo");
}

// ---------- números ----------

const $ = (id) => document.getElementById(id);
const locale = () => LOCALES[estado.idioma];

function num(valor, casas = 0, extra = {}) {
  return new Intl.NumberFormat(locale(), {
    minimumFractionDigits: casas, maximumFractionDigits: casas, ...extra,
  }).format(valor);
}

function pct(valor, casas = 1) {
  return new Intl.NumberFormat(locale(), {
    style: "percent", minimumFractionDigits: casas, maximumFractionDigits: casas,
  }).format(valor);
}

const volume = (milhoes, casas = 1) => t("fmt.volume", { valor: num(milhoes, casas) });
const lista = (itens) => new Intl.ListFormat(locale(), { type: "conjunction" }).format(itens);

// ---------- textos com números ----------

function preencher() {
  const { caso, modelo, benchmark: bench, oficial, limites, links } = estado.dados;
  const erroProjecao = num(Math.abs(oficial.roi - oficial.roi_projetado) * 100, 2);
  const unidadePp = t("fmt.pp", { valor: "" }).trim();

  $("kpi-auc").textContent = num(oficial.auc, 4);
  $("kpi-auc-sub").textContent = t("abertura.kpi_auc_sub");
  $("kpi-roi").textContent = pct(oficial.roi, 2);
  $("kpi-roi-sub").textContent = t("abertura.kpi_roi_sub", { meta: pct(oficial.meta_roi, 0) });
  $("kpi-erro").innerHTML = `${erroProjecao}<span class="kpi__unidade"> ${unidadePp}</span>`;
  $("kpi-erro-sub").textContent = t("abertura.kpi_erro_sub", {
    projetado: pct(oficial.roi_projetado, 2), realizado: pct(oficial.roi, 2),
  });

  $("problema-texto").innerHTML = t("problema.texto", {
    contratos: num(caso.contratos_A), inad: pct(caso.inadimplencia_A, 1),
    ltv: pct(caso.ltv_medio, 0), lgd: pct(caso.lgd_media_realizada, 0),
  });
  $("problema-bases").textContent = t("problema.bases", {
    a: num(caso.contratos_A), b: num(caso.contratos_B), c: num(caso.propostas_C),
  });
  preencherSafras(caso.inadimplencia_por_ano);

  $("vaz-auc").textContent = num(modelo.auc_vazamento, 2);
  $("vaz-teste").textContent = num(0);
  $("vaz-teste-rotulo").textContent = t("armadilha.teste", { n: num(caso.contratos_B + caso.propostas_C) });

  const candidato = (nome) => modelo.candidatos.find((c) => c.candidato === nome);
  const campeao = candidato(nomeDoCampeao());
  $("torneio-vencedor").innerHTML = t("torneio.vencedor", {
    ganho: num(candidato("ensemble").auc_medio - campeao.auc_medio, 3),
    auc: num(campeao.auc_J2, 3), ks: num(campeao.ks_J2, 3),
  });
  $("torneio-psi").innerHTML = t("torneio.psi", { psi: num(modelo.psi_2023_2024, 4) });

  const faixa = (score) => modelo.faixas.find((f) => f.score_1a10 === score);
  $("regua-texto").innerHTML = t("regua.texto", {
    s1: pct(faixa(1).inadimplencia_observada_J2, 1), s10: pct(faixa(10).inadimplencia_observada_J2, 1),
    previsto: pct(modelo.previsto_2024, 1), observado: pct(modelo.observado_2024, 1),
  });

  const data = (iso) => new Intl.DateTimeFormat(locale(), { dateStyle: "medium" }).format(new Date(`${iso}T12:00:00`));
  const [inicio, fim] = bench.instituicoes.periodo;
  $("mercado-grafico-sub").textContent = t("mercado.grafico_sub", {
    n: bench.instituicoes.quantidade, inicio: data(inicio), fim: data(fim),
  });
  $("mercado-texto").innerHTML = t("mercado.texto", {
    n: bench.instituicoes.quantidade,
    antiga: pct(bench.taxa_antiga_autocred / 100, 2),
    media: pct(bench.sgs_25471_juros_veiculos_pf.media_2022_2024 / 100, 2),
    min: pct(bench.politica_min / 100, 1), max: pct(bench.politica_max / 100, 1),
  });

  preencherPlacar(oficial, limites);
  $("resultado-texto").innerHTML = t("resultado.texto", {
    roi: pct(oficial.roi, 2), inad: pct(oficial.inadimplencia, 2),
  });
  $("resultado-projetado").innerHTML = t("resultado.projetado", {
    projetado: pct(oficial.roi_projetado, 2), realizado: pct(oficial.roi, 2),
    taxa: pct(oficial.taxa_media, 2), aceite: pct(oficial.aceite, 1),
  });
  $("licao1").innerHTML = t("resultado.licao1", {
    min: pct(bench.politica_min / 100, 1), max: pct(bench.politica_max / 100, 1),
  });
  $("licao2").innerHTML = t("resultado.licao2", {
    aprovacao: pct(oficial.aprovacao, 1), volume: volume(oficial.volume_mi, 1),
  });
  $("licao3").innerHTML = t("resultado.licao3", {
    erro: t("fmt.pp", { valor: erroProjecao }), projetado: pct(oficial.roi_projetado, 2), realizado: pct(oficial.roi, 2),
  });

  const consulta = new Date(`${bench.consulta_em}T12:00:00`);
  $("rodape-fontes").textContent = t("rodape.fontes", {
    data: new Intl.DateTimeFormat(locale(), { month: "long", year: "numeric" }).format(consulta),
  });
  $("link-codigo").href = links.repositorio;
  $("rodape-codigo").href = links.repositorio;
}

function preencherSafras(porAno) {
  const maximo = Math.max(...Object.values(porAno));
  $("safras").replaceChildren(...Object.entries(porAno).map(([ano, taxa]) => {
    const linha = document.createElement("div");
    linha.className = "safra";
    linha.innerHTML = `<span>${ano}</span><span class="safra__barra" style="width:${(taxa / maximo) * 100}%"></span>`
      + `<span class="safra__taxa">${pct(taxa, 1)}</span>`;
    return linha;
  }));
}

function preencherPlacar(oficial, limites) {
  $("res-roi").textContent = pct(oficial.roi, 2);
  $("res-roi-nota").textContent = t("resultado.roi_nota", {
    projetado: pct(oficial.roi_projetado, 2), meta: pct(oficial.meta_roi, 1),
  });
  $("res-inad").textContent = pct(oficial.inadimplencia, 2);
  $("res-inad-nota").textContent = t("resultado.inad_nota", { teto: pct(limites.inadimplencia_max, 1) });
  const [taxa, sufixo] = t("resultado.taxa_valor", { taxa: "\n" }).split("\n");
  $("res-taxa").innerHTML = `${taxa}${pct(oficial.taxa_media, 2)}<small>${sufixo.trim()}</small>`;
  $("res-taxa-nota").textContent = t("resultado.taxa_nota", { aceite: pct(oficial.aceite, 1) });

  // Cada medidor vai de zero a um pouco além do maior entre o valor e o limite; o traço dourado é o limite.
  const medidores = [
    { nome: t("resultado.lim_aprovacao"), valor: oficial.aprovacao, limite: limites.aprovacao_min, minimo: true, fmt: (v) => pct(v, 1) },
    { nome: t("resultado.lim_inad"), valor: oficial.inadimplencia, limite: limites.inadimplencia_max, minimo: false, fmt: (v) => pct(v, 2) },
    { nome: t("resultado.lim_volume"), valor: oficial.volume_mi, limite: limites.volume_min_mi, minimo: true, fmt: (v) => volume(v, 1) },
    { nome: t("resultado.lim_taxa"), valor: oficial.taxa_media, limite: limites.taxa_max_am, minimo: false, fmt: (v) => pct(v, 2) },
  ];
  $("medidores").replaceChildren(...medidores.map((m) => {
    const escalaMax = Math.max(m.valor, m.limite) * 1.25;
    const ok = m.minimo ? m.valor >= m.limite : m.valor <= m.limite;
    const regra = t(m.minimo ? "resultado.lim_minimo" : "resultado.lim_maximo", { valor: m.fmt(m.limite) });
    const item = document.createElement("li");
    item.className = "medidor";
    item.innerHTML = `<span class="medidor__nome">${m.nome}</span>`
      + `<span class="medidor__trilho" aria-hidden="true">`
      + `<span class="medidor__valor-barra" style="width:${(m.valor / escalaMax) * 100}%"></span>`
      + `<span class="medidor__limite" style="left:${(m.limite / escalaMax) * 100}%"></span></span>`
      + `<span class="medidor__num"><b>${m.fmt(m.valor)}${marcaLimite(ok)}</b><span>${regra}</span></span>`;
    return item;
  }));
}

function nomeDoCampeao() {
  const { descricao } = estado.dados.modelo;
  return descricao.tipo + (descricao.monotonico ? "_monotonico" : "");
}

// ---------- gráficos (SVG) ----------

function svgEl(nome, atributos = {}, conteudo) {
  const el = document.createElementNS(SVG_NS, nome);
  for (const [chave, valor] of Object.entries(atributos)) el.setAttribute(chave, valor);
  if (conteudo != null) el.textContent = conteudo;
  return el;
}

function novoSvg(alvo, largura, altura, rotulo) {
  const svg = svgEl("svg", {
    viewBox: `0 0 ${largura} ${altura}`, width: largura, height: altura, role: "img", "aria-label": rotulo,
  });
  alvo.replaceChildren(svg);
  return svg;
}

const escala = (d0, d1, r0, r1) => (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
const larguraDe = (alvo) => Math.max(300, Math.floor(alvo.clientWidth));

function graficoTorneio() {
  const alvo = $("grafico-torneio");
  const { candidatos } = estado.dados.modelo;
  const campeao = nomeDoCampeao();
  const aucCampeao = candidatos.find((c) => c.candidato === campeao).auc_medio;
  const ordenados = [...candidatos].sort((a, b) => b.auc_medio - a.auc_medio);
  const W = larguraDe(alvo);
  const estreito = W < 560;
  const linha = estreito ? 62 : 46;
  const margem = { esq: 8, dir: 56, topo: 4 };
  const lo = Math.floor((Math.min(...ordenados.map((c) => c.auc_medio)) - 0.01) * 50) / 50;
  const hi = Math.ceil((Math.max(...ordenados.map((c) => c.auc_medio)) + 0.005) * 50) / 50;
  const x = escala(lo, hi, margem.esq + 6, W - margem.dir);
  const baseEixo = margem.topo + ordenados.length * linha + 6;
  const svg = novoSvg(alvo, W, baseEixo + 40, t("torneio.grafico"));

  for (let tick = lo; tick <= hi + 1e-9; tick += 0.02) {
    svg.append(svgEl("line", { class: "grade", x1: x(tick), x2: x(tick), y1: margem.topo, y2: baseEixo }));
    svg.append(svgEl("text", { x: x(tick), y: baseEixo + 16, "text-anchor": "middle" }, num(tick, 2)));
  }
  svg.append(svgEl("text", { x: margem.esq, y: baseEixo + 36 }, t("torneio.eixo")));

  ordenados.forEach((c, i) => {
    const y0 = margem.topo + i * linha;
    const yPonto = y0 + (estreito ? 46 : 32);
    const nome = svgEl("text", { class: "t-nome", x: margem.esq, y: y0 + 15 });
    nome.append(svgEl("tspan", {}, t(`nome.${c.candidato}`)));
    let tag = null;
    if (c.candidato === campeao) {
      tag = svgEl("tspan", { class: "t-tag" }, t("torneio.campeao"));
    } else if (c.auc_medio > aucCampeao) {
      tag = svgEl("tspan", { class: "t-tag t-tag--neutra" },
        t("torneio.ganho_pequeno", { ganho: num(c.auc_medio - aucCampeao, 3) }));
    }
    if (tag) {
      if (estreito) { tag.setAttribute("x", margem.esq); tag.setAttribute("dy", 16); } else { tag.setAttribute("dx", 10); }
      nome.append(tag);
    }
    svg.append(nome);
    svg.append(svgEl("line", { class: "trilha", x1: x(lo), x2: x(hi), y1: yPonto, y2: yPonto }));
    const destaque = c.candidato === campeao;
    svg.append(svgEl("circle", { class: destaque ? "ponto ponto--campeao" : "ponto", cx: x(c.auc_medio), cy: yPonto, r: destaque ? 7 : 5.5 }));
    svg.append(svgEl("text", { class: destaque ? "t-forte" : "t-valor", x: x(c.auc_medio) + 12, y: yPonto + 4 }, num(c.auc_medio, 3)));
  });
}

function graficoColunas(alvo, itens, { rotulo, rotuloX, maximo, valores = true }) {
  const W = larguraDe(alvo);
  const H = 250;
  const margem = { esq: 40, dir: 6, topo: 22, base: 30 };
  const y = escala(0, maximo, H - margem.base, margem.topo);
  const banda = (W - margem.esq - margem.dir) / itens.length;
  const svg = novoSvg(alvo, W, rotuloX ? H + 22 : H, rotulo);
  for (let tick = 0; tick <= maximo + 1e-9; tick += 0.05) {
    svg.append(svgEl("line", { class: "grade", x1: margem.esq, x2: W - margem.dir, y1: y(tick), y2: y(tick) }));
    svg.append(svgEl("text", { x: margem.esq - 6, y: y(tick) + 4, "text-anchor": "end" }, pct(tick, 0)));
  }
  itens.forEach((item, i) => {
    const centro = margem.esq + banda * (i + 0.5);
    const largura = Math.min(banda * 0.66, 96);
    svg.append(svgEl("rect", {
      class: item.classe || "barra", x: centro - largura / 2, y: y(item.valor),
      width: largura, height: y(0) - y(item.valor), rx: 2,
    }));
    if (valores) {
      svg.append(svgEl("text", { class: "t-valor", x: centro, y: y(item.valor) - 6, "text-anchor": "middle" }, pct(item.valor, 1)));
    }
    if (item.ponto != null) {
      svg.append(svgEl("circle", { class: "ponto ponto--observado", cx: centro, cy: y(item.ponto), r: 5 }));
    }
    svg.append(svgEl("text", { x: centro, y: H - margem.base + 18, "text-anchor": "middle" }, item.rotulo));
  });
  if (rotuloX) svg.append(svgEl("text", { x: margem.esq, y: H + 18 }, rotuloX));
}

function graficoIdade() {
  const idade = estado.dados.modelo.inadimplencia_por_idade;
  const maximo = Math.ceil(Math.max(...idade.map((f) => f.taxa)) * 20) / 20;
  graficoColunas($("grafico-idade"), idade.map((f) => ({ rotulo: f.faixa, valor: f.taxa })),
    { rotulo: t("torneio.u_grafico"), maximo });
}

function graficoRegua() {
  const faixas = [...estado.dados.modelo.faixas].sort((a, b) => a.score_1a10 - b.score_1a10);
  const maior = Math.max(...faixas.flatMap((f) => [f.pd_media, f.inadimplencia_observada_J2]));
  graficoColunas($("grafico-regua"), faixas.map((f) => ({
    rotulo: String(f.score_1a10), valor: f.pd_media, ponto: f.inadimplencia_observada_J2, classe: "barra barra--prevista",
  })), { rotulo: t("regua.titulo"), rotuloX: t("regua.eixo"), maximo: Math.ceil(maior * 20) / 20, valores: false });
}

function graficoMercado() {
  const alvo = $("grafico-mercado");
  const bench = estado.dados.benchmark;
  const W = larguraDe(alvo);
  const raio = W < 560 ? 4 : 6;
  const margem = { esq: 12, dir: 12, topo: 44, base: 44 };
  const lo = 0.5;
  const hi = Math.ceil(bench.instituicoes.maxima * 2) / 2;
  const x = escala(lo, hi, margem.esq, W - margem.dir);

  // Enxame: cada instituição sobe até o primeiro nível livre, para os pontos não se cobrirem.
  const niveis = [];
  const pontos = [...bench.taxas_instituicoes].sort((a, b) => a.taxa - b.taxa).map((inst) => {
    const cx = x(inst.taxa);
    let nivel = 0;
    while ((niveis[nivel] ?? -Infinity) > cx - 2 * raio - 1.5) nivel += 1;
    niveis[nivel] = cx;
    return { cx, nivel, inst };
  });
  const alturaEnxame = niveis.length * (2 * raio + 1.5);
  const base = margem.topo + alturaEnxame + 8;
  const H = base + margem.base;
  const svg = novoSvg(alvo, W, H, t("mercado.titulo"));
  const yNivel = (nivel) => base - 8 - raio - nivel * (2 * raio + 1.5);

  svg.append(svgEl("rect", {
    class: "faixa", x: x(bench.politica_min), y: margem.topo - 6,
    width: x(bench.politica_max) - x(bench.politica_min), height: base - margem.topo + 6,
  }));
  for (let tick = lo; tick <= hi + 1e-9; tick += 0.5) {
    svg.append(svgEl("line", { class: "grade", x1: x(tick), x2: x(tick), y1: base, y2: base + 5 }));
    svg.append(svgEl("text", { x: x(tick), y: base + 19, "text-anchor": "middle" }, pct(tick / 100, 1)));
  }
  svg.append(svgEl("line", { class: "grade", x1: margem.esq, x2: W - margem.dir, y1: base, y2: base }));
  svg.append(svgEl("text", { x: margem.esq, y: base + 38 }, t("mercado.eixo")));

  for (const p of pontos) {
    const circulo = svgEl("circle", { class: "ponto ponto--instituicao", cx: p.cx, cy: yNivel(p.nivel), r: raio });
    circulo.append(svgEl("title", {}, `${p.inst.instituicao}: ${pct(p.inst.taxa / 100, 2)}`));
    svg.append(circulo);
  }

  const marcos = [
    { valor: bench.taxa_antiga_autocred, classe: "linha-antiga", texto: "t-alerta", y: 14 },
    { valor: bench.sgs_25471_juros_veiculos_pf.media_2022_2024, classe: "linha-media", texto: "t-valor", y: 30 },
  ];
  for (const m of marcos) {
    const cx = x(m.valor);
    svg.append(svgEl("line", { class: m.classe, x1: cx, x2: cx, y1: m.y + 4, y2: base }));
    const ancora = cx > W - 60 ? "end" : cx < 60 ? "start" : "middle";
    svg.append(svgEl("text", { class: m.texto, x: cx, y: m.y, "text-anchor": ancora }, pct(m.valor / 100, 2)));
  }
}

function desenharGraficos() {
  estado.largura = document.querySelector(".pagina").clientWidth;
  graficoTorneio();
  graficoIdade();
  graficoRegua();
  graficoMercado();
}

// ---------- simulador ----------

function lerChave(chave) {
  const [corte, ajuste, entrada, fora] = chave.split("|");
  return { corte: Number(corte), ajuste: Number(ajuste), entrada: Number(entrada) / 100, fora };
}

const chaveDe = (s) => `${s.corte}|${s.ajuste}|${Math.round(s.entrada * 100)}|${s.fora}`;

function segmentos(alvoId, nome, opcoes, atual, aoMudar) {
  $(alvoId).replaceChildren(...opcoes.map(({ valor, rotulo }) => {
    const label = document.createElement("label");
    label.className = "segmento";
    const input = document.createElement("input");
    input.type = "radio";
    input.name = nome;
    input.id = `${nome}-${String(valor).replace(".", "_")}`;
    input.value = String(valor);
    input.checked = valor === atual;
    input.addEventListener("change", () => aoMudar(valor));
    const span = document.createElement("span");
    span.textContent = rotulo;
    label.append(input, span);
    return label;
  }));
}

function montarControles() {
  const sim = estado.dados.simulador;
  const mudar = (campo) => (valor) => { estado.sim[campo] = valor; atualizarSimulador(); };
  segmentos("ctl-corte", "corte", sim.cortes.map((c) => ({ valor: c, rotulo: String(c) })), estado.sim.corte, mudar("corte"));
  segmentos("ctl-entrada", "entrada", sim.entradas.map((e) => ({ valor: e, rotulo: pct(e, 0) })), estado.sim.entrada, mudar("entrada"));
  segmentos("ctl-fora", "fora", sim.fora_perfil.map((f) => ({ valor: f, rotulo: t(`simulador.${f}`) })), estado.sim.fora, mudar("fora"));
  const deslizante = $("ctl-ajuste");
  deslizante.min = 0;
  deslizante.max = sim.ajustes_pb.length - 1;
  deslizante.step = 1;
  deslizante.value = sim.ajustes_pb.indexOf(estado.sim.ajuste);

  // Escala sob o controle: extremos e a posição do zero (a escada sem ajuste).
  const pontos = (pb) => num(pb / 100, 1, { signDisplay: "exceptZero" });
  $("ajuste-min").textContent = pontos(sim.ajustes_pb[0]);
  $("ajuste-max").textContent = pontos(sim.ajustes_pb.at(-1));
  $("ajuste-zero").textContent = pontos(0);
  $("ajuste-escala").style.setProperty("--zero", sim.ajustes_pb.indexOf(0) / (sim.ajustes_pb.length - 1));
}

function taxasDaEscada() {
  const { escada, teto_taxa_am: teto } = estado.dados.simulador;
  const taxas = {};
  for (let score = 1; score <= 10; score += 1) {
    taxas[score] = Math.min(escada[score] + estado.sim.ajuste / 10000, teto);
  }
  return taxas;
}

// Marca discreta ao lado do valor: ✓ quando cumpre o limite, ✕ quando fura. O texto vai para o leitor de tela.
function marcaLimite(ok) {
  const texto = t(ok ? "simulador.cumpre" : "simulador.fura");
  return `<span class="marca-limite ${ok ? "marca-limite--ok" : "marca-limite--fura"}" role="img"`
    + ` aria-label="${texto}" title="${texto}">${ok ? "✓" : "✕"}</span>`;
}

function metrica(rotulo, valor, ok) {
  return `<div class="metrica"><dt>${rotulo}</dt>`
    + `<dd class="metrica__valor${ok ? "" : " metrica__valor--fura"}">${valor}${marcaLimite(ok)}</dd></div>`;
}

function cartaoCenario(cenario, r, meta) {
  const cartao = document.createElement("article");
  const acima = r.roi > meta;
  cartao.className = cenario === "base" ? "cenario cenario--base" : "cenario";
  cartao.innerHTML = `<header><h3 class="cenario__nome">${t(`simulador.nome_${cenario}`)}</h3>`
    + `<p class="cenario__desc">${t(`simulador.cenario_${cenario}`)}</p></header>`
    + `<div class="cenario__roi"><span class="rotulo">${t("simulador.roi")}</span>`
    + `<strong class="cenario__roi-num">${pct(r.roi, 1)}</strong>`
    + `<span class="meta-status${acima ? "" : " meta-status--fura"}">`
    + `<span aria-hidden="true">${acima ? "✓" : "✕"}</span> ${t(acima ? "simulador.acima_meta" : "simulador.abaixo_meta")}</span></div>`
    + `<dl class="metricas">${metrica(t("simulador.aprovacao"), pct(r.aprovacao, 1), r.ok_aprovacao)}`
    + `${metrica(t("simulador.volume"), volume(r.volume_mi, 1), r.ok_volume)}`
    + `${metrica(t("simulador.inadimplencia"), pct(r.inadimplencia, 1), r.ok_inadimplencia)}</dl>`;
  return cartao;
}

function atualizarSimulador() {
  const sim = estado.dados.simulador;
  const oficial = estado.dados.oficial;
  const chave = chaveDe(estado.sim);
  const resultado = sim.resultados[chave];
  const taxas = taxasDaEscada();
  const aprovadas = Object.entries(taxas).filter(([score]) => Number(score) >= estado.sim.corte).map(([, taxa]) => taxa);

  $("saida-ajuste").textContent = t("simulador.taxa_valor", {
    ajuste: num(estado.sim.ajuste / 100, 1, { signDisplay: "exceptZero" }),
    min: pct(Math.min(...aprovadas), 1), max: pct(Math.max(...aprovadas), 1),
  });

  $("escada").replaceChildren(...Object.entries(taxas).map(([score, taxa]) => {
    const degrau = document.createElement("li");
    const aprova = Number(score) >= estado.sim.corte;
    degrau.className = aprova ? "degrau" : "degrau degrau--nega";
    degrau.innerHTML = `<span class="degrau__score">${score}</span>`
      + `<span class="degrau__taxa">${aprova ? pct(taxa, 1) : t("simulador.nega")}</span>`;
    return degrau;
  }));

  const quebras = CENARIOS.filter((c) => {
    const r = resultado[c];
    return !(r.ok_aprovacao && r.ok_inadimplencia && r.ok_volume);
  });
  const status = $("sim-status");
  status.className = quebras.length ? "status status--fura" : "status status--ok";
  status.textContent = quebras.length
    ? t("simulador.status_quebra", { lista: lista(quebras.map((c) => t(`simulador.nome_${c}`))) })
    : t("simulador.status_ok");

  const vencedora = chave === sim.vencedora;
  $("sim-vencedora").hidden = !vencedora;
  $("sim-vencedora").textContent = t("simulador.vencedora_badge", { roi: pct(oficial.roi, 2), volume: volume(oficial.volume_mi, 1) });
  $("btn-vencedora").setAttribute("aria-pressed", String(vencedora));

  $("cenarios").replaceChildren(...CENARIOS.map((c) => cartaoCenario(c, resultado[c], oficial.meta_roi)));
}

// ---------- montagem ----------

function render() {
  aplicarTextos();
  preencher();
  montarControles();
  atualizarSimulador();
  desenharGraficos();
}

function trocarIdioma() {
  estado.idioma = estado.idioma === "pt" ? "en" : "pt";
  salvarIdioma(estado.idioma);
  history.replaceState(null, "", `?lang=${estado.idioma}${location.hash}`);
  render();
}

async function carregar(arquivo) {
  const resposta = await fetch(arquivo, { cache: "no-cache" });
  if (!resposta.ok) throw new Error(`${arquivo}: HTTP ${resposta.status}`);
  return resposta.json();
}

async function iniciar() {
  estado.idioma = idiomaInicial();
  try {
    [estado.dados, estado.textos] = await Promise.all([carregar("dados.json"), carregar("textos.json")]);
  } catch (erro) {
    console.error(erro);
    $("erro-carga").hidden = false;
    return;
  }
  estado.sim = lerChave(estado.dados.simulador.vencedora);
  render();
  document.body.classList.add("pronto");

  $("idioma").addEventListener("click", trocarIdioma);
  $("btn-vencedora").addEventListener("click", () => {
    estado.sim = lerChave(estado.dados.simulador.vencedora);
    montarControles();
    atualizarSimulador();
  });
  $("ctl-ajuste").addEventListener("input", (evento) => {
    estado.sim.ajuste = estado.dados.simulador.ajustes_pb[Number(evento.target.value)];
    atualizarSimulador();
  });
  $("controles").addEventListener("submit", (evento) => evento.preventDefault());

  let espera;
  new ResizeObserver(() => {
    clearTimeout(espera);
    espera = setTimeout(() => {
      if (document.querySelector(".pagina").clientWidth !== estado.largura) desenharGraficos();
    }, 120);
  }).observe(document.querySelector(".pagina"));
}

iniciar();
